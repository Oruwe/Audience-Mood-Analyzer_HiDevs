#!/usr/bin/env python3
"""Recover archived rows into the live DuckDB warehouse.

Schema migrations archive the pre-upgrade table as
``analyzed_comments_legacy_<timestamp>``. This tool copies every archived row
that is not already present in the live ``analyzed_comments`` table, filling
columns that did not exist in the legacy layout with NULL (downstream code
already tolerates missing embeddings and cluster ids).

Stop the pipeline before running — DuckDB permits only one writer.

Usage (from repo root):
    python recover_legacy.py           # copy rows, keep the archives
    python recover_legacy.py --drop    # copy rows, then drop recovered archives

Exit code 0 = nothing to do, or every archive recovered cleanly;
exit code 1 = at least one archive could not be recovered.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import duckdb  # noqa: E402

from storage.db import DB_PATH, _COLUMNS  # noqa: E402


def _list_archive_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_name LIKE 'analyzed_comments_legacy%' "
        "ORDER BY table_name"
    ).fetchall()
    return [r[0] for r in rows]


def _present_columns(conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM duckdb_columns() "
            f"WHERE table_name = '{table}'"
        ).fetchall()
    }


def _recover_table(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    """Copy unseen rows out of *table*; return how many rows were inserted."""
    present = _present_columns(conn, table)
    if "comment_id" not in present:
        raise ValueError("archive has no comment_id column — cannot de-duplicate safely")

    select_parts = [f'l."{col}"' if col in present else "NULL" for col in _COLUMNS]
    sql = (
        f"INSERT INTO analyzed_comments "
        f"SELECT {', '.join(select_parts)} FROM {table} l "
        f"WHERE l.comment_id IS NOT NULL AND NOT EXISTS ("
        f"  SELECT 1 FROM analyzed_comments t WHERE t.comment_id = l.comment_id)"
    )
    return int(conn.execute(sql).fetchone()[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover legacy DuckDB archive rows")
    parser.add_argument("--drop", action="store_true",
                        help="drop each archive table after a successful copy")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"No warehouse at {DB_PATH} — nothing to recover.")
        return 0

    failures = 0
    with duckdb.connect(str(DB_PATH)) as conn:
        archives = _list_archive_tables(conn)
        if not archives:
            print("No analyzed_comments_legacy_* tables found — nothing to recover.")
            return 0

        for table in archives:
            try:
                inserted = _recover_table(conn, table)
                print(f"✅ {table}: recovered {inserted} row(s)")
                if args.drop:
                    conn.execute(f"DROP TABLE {table}")
                    print(f"   dropped {table}")
            except Exception as exc:  # noqa: BLE001 — report and keep going
                failures += 1
                print(f"❌ {table}: recovery failed ({type(exc).__name__}: {exc})")

    if failures == 0:
        print("\nDone.")
    else:
        print(f"\n{failures} archive(s) need manual attention.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
