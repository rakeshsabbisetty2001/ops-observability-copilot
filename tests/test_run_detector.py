import duckdb
import pytest

import app.config as config_module
from scripts.generate_data import generate, write_to_db
from scripts.run_detector import run_detector


@pytest.fixture
def temp_db_paths(tmp_path, monkeypatch):
    ops_path = str(tmp_path / "ops.duckdb")
    gt_path = str(tmp_path / "ground_truth.duckdb")
    monkeypatch.setattr(config_module.settings, "duckdb_path", ops_path)
    monkeypatch.setattr(config_module.settings, "ground_truth_duckdb_path", gt_path)
    return ops_path, gt_path


def test_run_detector_on_a_fresh_db_creates_all_tables(temp_db_paths):
    """Epic 3 review round 2, #3 — the round-1 write-path test always ran
    write_to_db() first, which already creates events/query_log itself, so
    it passed unchanged even with init_schema() deleted from run_detector.
    This test calls run_detector() with NO prior generator run, so init_schema
    is the only thing that could create events/query_log — it genuinely fails
    without that call (verified: reverting it raises CatalogException here)."""
    ops_path, _ = temp_db_paths
    result = run_detector()
    assert result.empty

    conn = duckdb.connect(ops_path, read_only=True)
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    assert tables == {"events", "detected_anomalies", "query_log"}
    conn.close()


def test_run_detector_writes_all_tables_and_preserves_events(temp_db_paths):
    """Epic 3 review round 1, #4 — this is the exact gap that let Epic 1-2's
    init_schema regression through undetected: a script's write path with no
    test guarding it. run_detector reuses the DROP+CREATE+INSERT pattern on a
    different table; nothing was checking it kept the same guarantees."""
    ops_path, _ = temp_db_paths
    events_df, gt_df = generate(seed=1, days=2, interval_minutes=15)
    write_to_db(events_df, gt_df)

    result = run_detector()

    conn = duckdb.connect(ops_path, read_only=True)
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    assert tables == {"events", "detected_anomalies", "query_log"}
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == len(events_df)
    assert conn.execute("SELECT COUNT(*) FROM detected_anomalies").fetchone()[0] == len(result)
    conn.close()


def test_run_detector_sample_event_ids_roundtrip(temp_db_paths):
    """BIGINT[] through pandas -> DuckDB -> pandas must come back as the same
    list of ints it went in as."""
    ops_path, _ = temp_db_paths
    events_df, gt_df = generate(seed=1, days=2, interval_minutes=15)
    write_to_db(events_df, gt_df)
    result = run_detector()
    if result.empty:
        pytest.skip("no detections at this small scale/seed")

    conn = duckdb.connect(ops_path, read_only=True)
    row = conn.execute("SELECT id, sample_event_ids FROM detected_anomalies ORDER BY id LIMIT 1").fetchone()
    conn.close()
    expected = result.sort_values("id").iloc[0]
    assert list(row[1]) == list(expected["sample_event_ids"])


def test_run_detector_is_idempotent(temp_db_paths):
    """A second run must replace, not accumulate."""
    ops_path, _ = temp_db_paths
    events_df, gt_df = generate(seed=1, days=2, interval_minutes=15)
    write_to_db(events_df, gt_df)

    result1 = run_detector()
    result2 = run_detector()
    assert len(result1) == len(result2)

    conn = duckdb.connect(ops_path, read_only=True)
    assert conn.execute("SELECT COUNT(*) FROM detected_anomalies").fetchone()[0] == len(result2)
    conn.close()
