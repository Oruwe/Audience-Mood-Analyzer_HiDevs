"""One-shot warehouse repair: round-trip every row through the hardened reader
(storage/db.py::_parse_jsonish) and rewrite via the canonical writer, purging
legacy non-JSON leftovers from the emotional_drivers / embedding columns.

Run from the repo root AFTER applying the hardened reader in storage/db.py:
    python scripts/maintenance/repair_warehouse.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import duckdb  # noqa: E402

import storage.db as db  # noqa: E402


def main() -> None:
    try:
        records = db.query_enriched_records()  # hardened reader tolerates the junk
    except FileNotFoundError:
        print(f"Nothing to repair — {db.DB_PATH} does not exist yet.")
        return
    print(f"Recovered {len(records)} records.")

    with duckdb.connect(str(db.DB_PATH)) as conn:
        conn.execute("DELETE FROM analyzed_comments")

    for record in records:
        db.insert_enriched_record(record)  # canonical JSON re-written cleanly
    print("Warehouse repaired.")


if __name__ == "__main__":
    main()
