"""L1 unit tests — ingestion.youtube quota arithmetic (SPEC §11 L1:
"quota arithmetic"), no network.
"""

import pytest

import ingestion.youtube as yt
from ingestion.youtube import QuotaExceededError, QuotaLedger, _comment_pages


@pytest.mark.parametrize("comment_count,expected_pages", [
    (0, 0),
    (1, 1),
    (99, 1),
    (100, 1),
    (101, 2),
    (250, 3),
    (10_000, 100),
])
def test_comment_pages_ceiling_division(comment_count, expected_pages):
    assert _comment_pages(comment_count) == expected_pages


def test_ledger_starts_with_full_budget():
    ledger = QuotaLedger(daily_budget=1000)
    assert ledger.spent == 0
    assert ledger.remaining == 1000
    assert ledger.can_afford(1000)
    assert not ledger.can_afford(1001)


def test_ledger_charge_reduces_remaining():
    ledger = QuotaLedger(daily_budget=100)
    ledger.charge(30)
    assert ledger.spent == 30
    assert ledger.remaining == 70


def test_ledger_require_raises_with_useful_numbers_when_unaffordable():
    ledger = QuotaLedger(daily_budget=10)
    ledger.charge(8)
    with pytest.raises(QuotaExceededError) as exc_info:
        ledger.require(5)
    err = exc_info.value
    assert err.required_units == 5
    assert err.remaining_units == 2
    assert "5" in str(err) and "2" in str(err)


def test_ledger_require_does_not_charge_on_refusal():
    ledger = QuotaLedger(daily_budget=10)
    ledger.charge(8)
    with pytest.raises(QuotaExceededError):
        ledger.require(5)
    assert ledger.spent == 8  # refused request must not itself cost anything


def test_ledger_rolls_over_on_a_new_pacific_day(monkeypatch):
    ledger = QuotaLedger(daily_budget=100)
    ledger.charge(90)
    assert ledger.remaining == 10

    monkeypatch.setattr(yt, "_current_quota_day", lambda: "2099-01-01")
    assert ledger.remaining == 100  # new day -> fresh budget
    assert ledger.spent == 0
