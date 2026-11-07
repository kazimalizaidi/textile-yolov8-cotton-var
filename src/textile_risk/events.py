"""Module 2 - Event-driven automation: broker, severity detector, supplier hedging."""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple

import numpy as np

from .common import atomic_write_json, utc_now_iso

LOGGER = logging.getLogger("textile.events")


class EventType(str, Enum):
    DEFECT_DETECTED = "defect.detected"
    CRITICAL_DEFECT = "defect.critical"
    HEDGE_EXECUTED = "hedge.executed"


@dataclass(frozen=True)
class Event:
    event_type: EventType
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=utc_now_iso)


EventHandler = Callable[[Event], None]


class InMemoryEventBroker:
    """
    Redis-style broker: a FIFO list (LPUSH/BRPOP semantics) drained by worker threads,
    topic fan-out to subscribers, exponential-backoff retries and a dead-letter list.
    """

    _STOP = object()

    def __init__(self, num_workers: int = 3, max_retries: int = 2, backoff_s: float = 0.05) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._workers: List[threading.Thread] = []
        self._num_workers = num_workers
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._lock = threading.Lock()
        self._stats: Counter[str] = Counter()
        self.dead_letters: List[Tuple[Event, str, str]] = []

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: Event) -> None:
        with self._lock:
            self._stats[f"published:{event.event_type.value}"] += 1
        self._queue.put(event)

    def start(self) -> None:
        for i in range(self._num_workers):
            t = threading.Thread(target=self._worker_loop, name=f"worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)
        LOGGER.info("Event broker started with %d workers", self._num_workers)

    def drain(self) -> None:
        self._queue.join()

    def stop(self) -> None:
        self.drain()
        for _ in self._workers:
            self._queue.put(self._STOP)
        for t in self._workers:
            t.join(timeout=5.0)
        self._workers.clear()
        LOGGER.info("Event broker stopped. Stats: %s", dict(self._stats))

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                self._dispatch(item)
            finally:
                self._queue.task_done()

    def _dispatch(self, event: Event) -> None:
        for handler in self._handlers.get(event.event_type, []):
            name = getattr(handler, "__qualname__", repr(handler))
            for attempt in range(self._max_retries + 1):
                try:
                    handler(event)
                    with self._lock:
                        self._stats[f"handled:{event.event_type.value}"] += 1
                    break
                except Exception as exc:  # noqa: BLE001 - worker must never die
                    LOGGER.warning("Handler %s failed on %s (attempt %d/%d): %s", name,
                                   event.event_id, attempt + 1, self._max_retries + 1, exc)
                    if attempt == self._max_retries:
                        with self._lock:
                            self._stats["dead_lettered"] += 1
                            self.dead_letters.append((event, name, repr(exc)))
                        LOGGER.error("Event %s dead-lettered for handler %s", event.event_id, name)
                    else:
                        time.sleep(self._backoff_s * (2 ** attempt))


@dataclass
class _BatchStreamState:
    next_seq: int = 0
    pending: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    streak: List[Dict[str, Any]] = field(default_factory=list)


class ConsecutiveSevereDefectDetector:
    """
    Fires CRITICAL_DEFECT after `threshold` consecutive severe detections within a batch.
    Events carry a per-batch sequence number and are re-ordered before evaluation, so the
    "consecutive" semantics hold even when several workers process events concurrently.
    A non-severe detection breaks the streak; the streak resets after each trigger.
    """

    def __init__(self, broker: InMemoryEventBroker, threshold: int = 3) -> None:
        if threshold < 2:
            raise ValueError("threshold must be >= 2")
        self._broker = broker
        self._threshold = threshold
        self._states: Dict[str, _BatchStreamState] = {}
        self._lock = threading.Lock()

    def handle(self, event: Event) -> None:
        payload = event.payload
        triggers: List[Dict[str, Any]] = []
        with self._lock:
            state = self._states.setdefault(payload["batch_id"], _BatchStreamState())
            state.pending[int(payload["seq"])] = payload
            while state.next_seq in state.pending:
                current = state.pending.pop(state.next_seq)
                state.next_seq += 1
                if current["is_severe"]:
                    state.streak.append(current)
                    if len(state.streak) == self._threshold:
                        triggers.append({
                            "batch_id": current["batch_id"],
                            "threshold": self._threshold,
                            "frame_ids": [p["frame_id"] for p in state.streak],
                            "severities": [round(p["severity"], 4) for p in state.streak],
                            "mean_severity": round(float(np.mean([p["severity"] for p in state.streak])), 4),
                            "trigger_seq": current["seq"],
                        })
                        state.streak = []
                else:
                    state.streak = []
        for trig in triggers:
            LOGGER.critical("CRITICAL: %d consecutive severe defects in batch %s (frames %s)",
                            trig["threshold"], trig["batch_id"], trig["frame_ids"])
            self._broker.publish(Event(EventType.CRITICAL_DEFECT, trig))


@dataclass(frozen=True)
class Supplier:
    supplier_id: str
    name: str
    material: str
    base_price_per_kg_usd: float
    lead_time_days: int
    reliability: float  # probability of on-time delivery in (0, 1]
    endpoint: str


@dataclass(frozen=True)
class HedgeConfig:
    material: str = "cotton_lint"
    cover_length_m: float = 5000.0  # production length to be covered by alternative supply
    fabric_gsm: float = 180.0
    fabric_width_m: float = 1.6
    yield_fraction: float = 0.92
    downtime_cost_per_day_usd: float = 12_000.0
    urgency_premium: float = 0.06


class MockSupplierApiClient:
    """Deterministic mock of a supplier RFQ REST API (no network I/O)."""

    def request_quote(self, supplier: Supplier, request: Dict[str, Any]) -> Dict[str, Any]:
        seed = int(hashlib.sha256(
            (supplier.supplier_id + request["headers"]["Idempotency-Key"]).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        time.sleep(0.01)  # emulate network latency
        quote = supplier.base_price_per_kg_usd * (1.0 + request["body"]["urgency_premium"]) \
            * float(rng.uniform(0.98, 1.05))
        return {
            "status_code": 201,
            "supplier_id": supplier.supplier_id,
            "quote_id": f"Q-{int(rng.integers(0, 16**12)):012x}",
            "unit_price_usd_per_kg": round(float(quote), 4),
            "lead_time_days": supplier.lead_time_days + int(rng.integers(0, 3)),
            "valid_until": utc_now_iso(),
            "latency_ms": 10,
        }


class SupplierHedgeOrchestrator:
    """Consumes CRITICAL_DEFECT events, requests quotes from all alternative suppliers,
    selects the best one and persists the full request/response payload as JSON."""

    def __init__(self, broker: InMemoryEventBroker, suppliers: Sequence[Supplier],
                 client: MockSupplierApiClient, config: HedgeConfig, output_dir: Path) -> None:
        self._broker = broker
        self._suppliers = list(suppliers)
        self._client = client
        self._cfg = config
        self._dir = output_dir
        self._lock = threading.Lock()
        self._hedged_batches: Set[str] = set()
        self.executed_files: List[Path] = []

    def _required_kg(self) -> float:
        fabric_kg = self._cfg.cover_length_m * self._cfg.fabric_width_m * self._cfg.fabric_gsm / 1000.0
        return fabric_kg / self._cfg.yield_fraction

    def handle(self, event: Event) -> None:
        batch_id = str(event.payload["batch_id"])
        with self._lock:  # idempotency: one hedge per batch
            if batch_id in self._hedged_batches:
                LOGGER.info("Hedge already executed for batch %s; suppressing event %s",
                            batch_id, event.event_id)
                return
            self._hedged_batches.add(batch_id)

        quantity = round(self._required_kg(), 2)
        idem = hashlib.sha256(f"{batch_id}:{event.event_id}".encode()).hexdigest()[:32]
        quotes: List[Dict[str, Any]] = []
        for sup in self._suppliers:
            request = {
                "method": "POST", "url": f"{sup.endpoint}/v1/rfq",
                "headers": {"Content-Type": "application/json", "Idempotency-Key": idem},
                "body": {"material": sup.material, "quantity_kg": quantity,
                         "urgency_premium": self._cfg.urgency_premium,
                         "trigger_event_id": event.event_id, "batch_id": batch_id},
            }
            response = self._client.request_quote(sup, request)
            expected_cost = (quantity * response["unit_price_usd_per_kg"]
                             + response["lead_time_days"] * self._cfg.downtime_cost_per_day_usd)
            quotes.append({"supplier": asdict(sup), "request": request, "response": response,
                           "risk_adjusted_cost_usd": round(expected_cost / sup.reliability, 2)})
        best = min(quotes, key=lambda q: q["risk_adjusted_cost_usd"])
        artefact = {
            "hedge_id": f"HEDGE-{uuid.uuid4().hex[:10].upper()}",
            "created_at": utc_now_iso(),
            "trigger_event": {"event_id": event.event_id, "type": event.event_type.value,
                              "payload": event.payload},
            "selection_rule": "min( (qty*price + lead_days*downtime_cost) / reliability )",
            "selected_supplier_id": best["supplier"]["supplier_id"],
            "selected_quote_id": best["response"]["quote_id"],
            "quotes": quotes,
        }
        path = self._dir / f"hedge_{batch_id}_{event.event_id[:8]}.json"
        atomic_write_json(path, artefact)
        with self._lock:
            self.executed_files.append(path)
        LOGGER.warning("Hedge %s executed for batch %s -> %s (risk-adjusted $%.2f) payload=%s",
                       artefact["hedge_id"], batch_id, best["supplier"]["name"],
                       best["risk_adjusted_cost_usd"], path)
        self._broker.publish(Event(EventType.HEDGE_EXECUTED,
                                   {"batch_id": batch_id, "hedge_id": artefact["hedge_id"],
                                    "file": str(path)}))

