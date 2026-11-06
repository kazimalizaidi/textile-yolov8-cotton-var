import json
import threading
from typing import List

from textile_risk.events import (ConsecutiveSevereDefectDetector, Event, EventType, HedgeConfig,
                                 InMemoryEventBroker, MockSupplierApiClient,
                                 SupplierHedgeOrchestrator)
from textile_risk.orchestrator import DEFAULT_SUPPLIERS


def _detected(batch: str, seq: int, severe: bool, frame: int = 0) -> Event:
    return Event(EventType.DEFECT_DETECTED, {"batch_id": batch, "seq": seq, "frame_id": frame,
                                              "severity": 3.0 if severe else 1.0, "is_severe": severe})


def _collect(broker: InMemoryEventBroker) -> List[Event]:
    out: List[Event] = []
    lock = threading.Lock()

    def sink(e: Event) -> None:
        with lock:
            out.append(e)

    broker.subscribe(EventType.CRITICAL_DEFECT, sink)
    return out


def test_broker_delivers_to_all_subscribers():
    broker = InMemoryEventBroker(num_workers=2)
    a, b = [], []
    broker.subscribe(EventType.DEFECT_DETECTED, a.append)
    broker.subscribe(EventType.DEFECT_DETECTED, b.append)
    broker.start()
    for i in range(20):
        broker.publish(_detected("B", i, False))
    broker.stop()
    assert len(a) == len(b) == 20
    assert broker.stats["published:defect.detected"] == 20


def test_broker_retries_then_dead_letters():
    attempts = {"n": 0}

    def always_fail(_: Event) -> None:
        attempts["n"] += 1
        raise RuntimeError("boom")

    broker = InMemoryEventBroker(num_workers=1, max_retries=2, backoff_s=0.001)
    broker.subscribe(EventType.DEFECT_DETECTED, always_fail)
    broker.start()
    broker.publish(_detected("B", 0, False))
    broker.stop()
    assert attempts["n"] == 3
    assert len(broker.dead_letters) == 1 and broker.stats["dead_lettered"] == 1


def test_three_consecutive_severe_triggers_once_even_out_of_order():
    broker = InMemoryEventBroker(num_workers=1)
    crit = _collect(broker)
    det = ConsecutiveSevereDefectDetector(broker, threshold=3)
    broker.start()
    for seq in (2, 0, 1):  # arrival order scrambled by concurrency
        det.handle(_detected("B", seq, True, frame=seq))
    broker.stop()
    assert len(crit) == 1
    assert crit[0].payload["frame_ids"] == [0, 1, 2]


def test_non_severe_breaks_streak_and_streak_resets_after_trigger():
    broker = InMemoryEventBroker(num_workers=1)
    crit = _collect(broker)
    det = ConsecutiveSevereDefectDetector(broker, threshold=3)
    broker.start()
    pattern = [True, True, False, True, True, True, True, True, True]  # triggers at idx 5 and 8
    for i, sev in enumerate(pattern):
        det.handle(_detected("B", i, sev, frame=i))
    broker.stop()
    assert [c.payload["trigger_seq"] for c in crit] == [5, 8]


def test_streaks_are_isolated_per_batch():
    broker = InMemoryEventBroker(num_workers=1)
    crit = _collect(broker)
    det = ConsecutiveSevereDefectDetector(broker, threshold=3)
    broker.start()
    for i in range(2):
        det.handle(_detected("A", i, True))
        det.handle(_detected("B", i, True))
    broker.stop()
    assert crit == []


def test_hedge_orchestrator_writes_payload_picks_min_cost_and_is_idempotent(tmp_path):
    broker = InMemoryEventBroker(num_workers=1)
    hedger = SupplierHedgeOrchestrator(broker, DEFAULT_SUPPLIERS, MockSupplierApiClient(),
                                       HedgeConfig(), tmp_path)
    broker.start()
    ev = Event(EventType.CRITICAL_DEFECT, {"batch_id": "B-9", "frame_ids": [1, 2, 3]})
    hedger.handle(ev)
    hedger.handle(Event(EventType.CRITICAL_DEFECT, {"batch_id": "B-9", "frame_ids": [7, 8, 9]}))
    broker.stop()

    files = list(tmp_path.glob("hedge_B-9_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert len(data["quotes"]) == len(DEFAULT_SUPPLIERS)
    best = min(data["quotes"], key=lambda q: q["risk_adjusted_cost_usd"])
    assert data["selected_supplier_id"] == best["supplier"]["supplier_id"]
    assert data["trigger_event"]["payload"]["batch_id"] == "B-9"
    for q in data["quotes"]:
        assert q["request"]["method"] == "POST" and "Idempotency-Key" in q["request"]["headers"]


def test_mock_client_is_deterministic():
    sup = DEFAULT_SUPPLIERS[0]
    req = {"headers": {"Idempotency-Key": "k"}, "body": {"urgency_premium": 0.06}}
    r1, r2 = (MockSupplierApiClient().request_quote(sup, req) for _ in range(2))
    assert r1["unit_price_usd_per_kg"] == r2["unit_price_usd_per_kg"]
    assert r1["quote_id"] == r2["quote_id"]
