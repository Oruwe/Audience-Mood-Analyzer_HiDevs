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
    """The underlying page arithmetic, measured with the per-video cap lifted
    out of the way. This is the pre-cap contract and it still holds: the cap
    changes how many comments we ask for, not how pages are counted."""
    assert _comment_pages(comment_count, max_comments=10_000) == expected_pages


@pytest.mark.parametrize("comment_count,expected_pages", [
    (0, 0),
    (1, 1),
    (70, 1),
    (100, 1),
    (101, 1),
    (10_000, 1),
    (1_000_000, 1),
])
def test_comment_pages_is_bounded_by_the_per_video_cap(comment_count, expected_pages):
    """At the default cap the estimate must never exceed one page per video.

    This is the arithmetic half of the bound: `fetch_video_comments` stops at
    MAX_COMMENTS_PER_VIDEO_DEFAULT, so an estimate that still billed a
    100k-comment video at 1,000 units would refuse analyses the real pull
    could afford a thousand times over.
    """
    assert _comment_pages(comment_count) == expected_pages


def test_comment_pages_cap_never_exceeds_uncapped_cost():
    """Capping may only ever reduce the estimate, never inflate it -- a video
    with 3 comments costs one page whether or not a 70-cap is in force."""
    for count in (0, 1, 3, 70, 99, 100, 101, 5_000):
        assert _comment_pages(count) <= _comment_pages(count, max_comments=10_000)


def test_comment_pages_with_zero_cap_costs_nothing():
    """A cap of 0 means "fetch nothing", so it must also mean "charge
    nothing" -- otherwise the estimate reserves quota the pull never spends."""
    assert _comment_pages(10_000, max_comments=0) == 0


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
