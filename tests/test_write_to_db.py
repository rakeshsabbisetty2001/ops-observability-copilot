import duckdb
import pytest

import app.config as config_module
import app.db as db_module
from scripts.generate_data import generate, write_to_db


@pytest.fixture
def temp_db_paths(tmp_path, monkeypatch):
    ops_path = str(tmp_path / "ops.duckdb")
    gt_path = str(tmp_path / "ground_truth.duckdb")
    monkeypatch.setattr(config_module.settings, "duckdb_path", ops_path)
    monkeypatch.setattr(config_module.settings, "ground_truth_duckdb_path", gt_path)
    return ops_path, gt_path


def test_write_to_db_roundtrip(temp_db_paths):
    ops_path, gt_path = temp_db_paths
    events_df, gt_df = generate(seed=1, days=2, interval_minutes=15)
    write_to_db(events_df, gt_df)

    conn = duckdb.connect(ops_path, read_only=True)
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == len(events_df)
    row = conn.execute("SELECT id, service, metric_name, value, level FROM events ORDER BY id LIMIT 1").fetchone()
    expected = events_df.sort_values("id").iloc[0]
    assert row == (expected["id"], expected["service"], expected["metric_name"], expected["value"], expected["level"])
    conn.close()

    gt_conn = duckdb.connect(gt_path, read_only=True)
    assert gt_conn.execute("SELECT COUNT(*) FROM ground_truth_anomalies").fetchone()[0] == len(gt_df)
    gt_conn.close()


def test_ground_truth_not_in_ops_db(temp_db_paths):
    """The physical split from Epic 1-2 review #2 — the API's own DB file must
    never contain the answer key, regardless of what any guardrail does."""
    ops_path, _ = temp_db_paths
    events_df, gt_df = generate(seed=1, days=2, interval_minutes=15)
    write_to_db(events_df, gt_df)

    conn = duckdb.connect(ops_path, read_only=True)
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    assert "ground_truth_anomalies" not in tables
    conn.close()


def test_ops_db_has_all_its_own_tables(temp_db_paths):
    """Round 2 review, #1: a targeted DROP+CREATE of just `events` silently
    dropped init_schema() from write_to_db, which deleted detected_anomalies
    and query_log from the shipped corpus — the prior test only asserted what
    must NOT be there, never what must."""
    ops_path, _ = temp_db_paths
    events_df, gt_df = generate(seed=1, days=2, interval_minutes=15)
    write_to_db(events_df, gt_df)

    conn = duckdb.connect(ops_path, read_only=True)
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    assert tables == {"events", "detected_anomalies"}
    conn.close()


def test_read_only_blocks_external_access(temp_db_paths):
    """Epic 1-2 review #1: read_only alone doesn't stop COPY TO / read_csv_auto
    — enable_external_access=false must be set for the API's connection mode."""
    _ = temp_db_paths  # ensures we never touch the real production data/ops.duckdb
    events_df, gt_df = generate(seed=1, days=2, interval_minutes=15)
    write_to_db(events_df, gt_df)

    conn = db_module.get_connection(read_only=True)
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == len(events_df)

    with pytest.raises(duckdb.PermissionException):
        conn.execute("COPY (SELECT 1) TO 'should_not_be_written.csv'")
    with pytest.raises(duckdb.PermissionException):
        conn.execute("SELECT * FROM read_csv_auto('nonexistent.csv')")
    conn.close()
