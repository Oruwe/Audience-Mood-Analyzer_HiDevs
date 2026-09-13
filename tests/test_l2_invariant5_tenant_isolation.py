"""L2 contract test — SPEC §10 invariant 5 (scaffold, expected red/skipped).

"Tenant isolation — every query scoped by user_id, parameterized."
Proven by: "Integration test: user A's session cannot read user B's row."

Not implementable yet: this build's 9-phase plan has no Postgres/auth phase
(the phases run Cuts -> config+harness -> ingestion -> Stage A -> Stage B ->
orchestration -> Stage C+insights -> UI -> ship; SPEC §7's Postgres/Neon
persistence and §7's st.login auth aren't scheduled in it). There is no
user_id-scoped storage layer to test against yet.

This is intentionally left red-by-skip rather than deleted, so the gap is
visible in `pytest` output rather than silently absent: SPEC §10 treats this
invariant as required before the product handles real user data, and §6's
Definition of Done doesn't currently list it either. Flag before shipping.
"""

import pytest


@pytest.mark.skip(
    reason=(
        "No persistence/auth layer exists yet — SPEC §7 Postgres+st.login "
        "and the user_id-scoped storage this invariant needs aren't in the "
        "current 9-phase plan. Un-skip once that lands."
    )
)
def test_user_a_session_cannot_read_user_b_row():
    raise NotImplementedError
