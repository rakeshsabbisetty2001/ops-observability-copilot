from datetime import datetime, timedelta

import duckdb
from fastapi.testclient import TestClient

import app.config as config_module
from app.main import _ANOMALIES_MAX_LIMIT, _DRILLDOWN_MAX_PAD, _DRILLDOWN_MIN_PAD, _drilldown_pad, app
from scripts.generate_data import generate, write_to_db
from scripts.run_detector import run_detector

client = TestClient(app)


def _seed(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module.settings, "duckdb_path", str(tmp_path / "ops.duckdb"))
    monkeypatch.setattr(config_module.settings, "ground_truth_duckdb_path", str(tmp_path / "gt.duckdb"))
    monkeypatch.setattr(config_module.settings, "query_log_duckdb_path", str(tmp_path / "query_log.duckdb"))
    events_df, gt_df = generate(seed=1, days=3, interval_minutes=15)
    write_to_db(events_df, gt_df)
    run_detector()


def test_list_anomalies_returns_seeded_rows(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    response = client.get("/anomalies")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) > 0
    assert {"id", "service", "metric_name", "start_ts", "end_ts", "method", "score"} <= body[0].keys()


def test_list_anomalies_filters_by_service(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    all_rows = client.get("/anomalies").json()
    service = all_rows[0]["service"]
    filtered = client.get("/anomalies", params={"service": service}).json()
    assert len(filtered) > 0
    assert all(r["service"] == service for r in filtered)
    # A filter that actually filters, not one that happens to match
    # everything (Epic 7 review round 1, High #1, M5 already caught this
    # one — kept as the positive case the metric/since tests mirror below).
    if len(filtered) < len(all_rows):
        assert any(r["service"] != service for r in all_rows)


def test_list_anomalies_filters_by_metric(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    all_rows = client.get("/anomalies").json()
    metric = all_rows[0]["metric_name"]
    filtered = client.get("/anomalies", params={"metric": metric}).json()
    assert len(filtered) > 0
    assert all(r["metric_name"] == metric for r in filtered)
    # Deleting the metric filter branch entirely left this endpoint's old
    # test suite green (Epic 7 review round 1, High #1, M1) — assert the
    # filter actually excludes something rather than just matching a shape.
    assert len(filtered) < len(all_rows)


def test_list_anomalies_filters_by_since(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    all_rows = client.get("/anomalies").json()
    assert len(all_rows) >= 2
    starts = sorted(r["start_ts"] for r in all_rows)
    median_since = starts[len(starts) // 2]

    filtered = client.get("/anomalies", params={"since": median_since}).json()
    assert len(filtered) > 0
    assert all(r["start_ts"] >= median_since for r in filtered)
    # Inverting the comparison or deleting the branch both left the old
    # suite green (Epic 7 review round 1, High #1, M2/M3) — a strictly
    # smaller filtered count is what actually proves the filter ran.
    assert len(filtered) < len(all_rows)
    # Inclusive at the boundary — `since` equal to a row's own start_ts
    # must still match it. Untested previously: a `>` off-by-one still
    # passed every assertion above (Epic 7 review round 4, Low #3, Q2).
    assert any(r["start_ts"] == median_since for r in filtered)


def test_list_anomalies_limit_is_applied(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    all_rows = client.get("/anomalies").json()
    assert len(all_rows) > 1
    # Binding the constant instead of the `limit` param left this endpoint's
    # 422-edge validation green while the value never reached the query
    # (Epic 7 review round 2, High #1, N1).
    limited = client.get("/anomalies", params={"limit": 1}).json()
    assert len(limited) == 1
    assert limited[0] == all_rows[0]  # still newest-first, just capped


def test_list_anomalies_limit_upper_bound_enforced(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    # The `ge=1` half of the bound was exercised by the limit=1 test above;
    # nothing checked `le=_ANOMALIES_MAX_LIMIT`, so a client could request
    # an arbitrarily large limit with the suite green (Epic 7 review round
    # 3, High #1, O8).
    response = client.get("/anomalies", params={"limit": 10**9})
    assert response.status_code == 422
    # Pinned against a literal, not just "somewhere below a billion" — the
    # 10**9 case alone let _ANOMALIES_MAX_LIMIT itself drift from 500 to
    # 500000 with the suite green (Epic 7 review round 4, Low #2).
    assert _ANOMALIES_MAX_LIMIT == 500
    assert client.get("/anomalies", params={"limit": 501}).status_code == 422


def test_list_anomalies_empty_string_filter_is_ignored(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    all_rows = client.get("/anomalies").json()
    # `service=` on the wire should behave like the filter wasn't sent at
    # all, matching the UI's own `if service_filter:` guard — reverting to
    # `if service is not None:` left the suite green (Epic 7 review round
    # 2, High #1, N7).
    response = client.get("/anomalies", params={"service": "", "metric": ""}).json()
    assert response == all_rows


def test_list_anomalies_since_normalizes_timezone(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    all_rows = client.get("/anomalies").json()
    naive_since_str = sorted(r["start_ts"] for r in all_rows)[len(all_rows) // 2]
    naive_since_dt = datetime.fromisoformat(naive_since_str)

    # An aware string representing the exact same UTC instant, spelled with
    # a non-zero offset — if UTC-normalization runs, this must return the
    # identical row set as the plain naive query above. Deleting the
    # normalization branch leaves the tz-aware value to DuckDB's own
    # (inconsistent, per round 1's nit N4 measurement) aware/naive
    # comparison, which does not agree with it (Epic 7 review round 2,
    # High #1, N5).
    aware_since_dt = naive_since_dt + timedelta(hours=5)
    aware_since_str = aware_since_dt.isoformat() + "+05:00"

    naive_result = client.get("/anomalies", params={"since": naive_since_str}).json()
    aware_result = client.get("/anomalies", params={"since": aware_since_str}).json()
    assert aware_result == naive_result


def test_list_anomalies_ordered_newest_first(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    body = client.get("/anomalies").json()
    starts = [r["start_ts"] for r in body]
    assert starts == sorted(starts, reverse=True)


def test_list_anomalies_rejects_injection_attempt(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    before = client.get("/anomalies").json()
    response = client.get("/anomalies", params={"service": "' OR '1'='1"})
    assert response.status_code == 200
    assert response.json() == []  # no row's service is literally that string
    after = client.get("/anomalies").json()
    assert after == before  # nothing was executed/altered


def test_drilldown_pad_floor_and_ceiling():
    # Direct, corpus-independent pin on the constants an end-to-end test
    # can't reliably catch a shrunk floor for (Epic 7 review round 2, nit
    # N10: an end-to-end "context exists on both sides" assertion still
    # passes when the floor is weakened to ~0, since pad then just equals
    # the window's own width).
    #
    # Asserted against LITERALS, not against _DRILLDOWN_MIN_PAD/_MAX_PAD
    # themselves — comparing a constant to itself only proves "floor > 1
    # minute" / "ceiling < 1 day" and lets the actual value drift by 15x
    # with this test green (Epic 7 review round 3, High #1, O3/O4).
    assert _drilldown_pad(timedelta(minutes=1)) == timedelta(minutes=30)
    assert _drilldown_pad(timedelta(hours=1)) == timedelta(hours=1)  # between floor and ceiling: passes through
    assert _drilldown_pad(timedelta(days=1)) == timedelta(hours=6)
    # Sanity: the literals above must actually match the constants in use,
    # so a deliberate constant change updates this test rather than the
    # test silently testing a value the code no longer has.
    assert _DRILLDOWN_MIN_PAD == timedelta(minutes=30)
    assert _DRILLDOWN_MAX_PAD == timedelta(hours=6)


def test_get_anomaly_detail_includes_event_window(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    # The WIDEST anomaly, not just [0] — a fixed argument (e.g. calling
    # _drilldown_pad with _DRILLDOWN_MIN_PAD instead of the anomaly's own
    # width) can pass for a narrow anomaly whose real width is already
    # near the floor, and only shows up as a real behavioural difference
    # once the window is wide enough that the two diverge (Epic 7 review
    # round 3, High #1, O2).
    all_rows = client.get("/anomalies").json()
    anomaly = max(
        all_rows,
        key=lambda r: datetime.fromisoformat(r["end_ts"]) - datetime.fromisoformat(r["start_ts"]),
    )
    response = client.get(f"/anomalies/{anomaly['id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == anomaly["id"]
    assert "events" in body
    assert len(body["events"]) > 0
    assert {"ts", "value", "in_window"} <= body["events"][0].keys()

    # The event window must stay within its own service/metric: dropping
    # the metric_name filter on this query left every existing test green
    # while the chart could plot an unrelated metric interleaved with the
    # flagged one (Epic 7 review round 1, High #1, M7 — the most serious
    # mutation found).
    # Keyed on (ts -> value) rather than just ts: the synthetic generator
    # shares one time grid across every service/metric, so a ts-only check
    # can't tell a correctly-filtered event apart from a same-timestamp
    # event belonging to a different metric — comparing values closes that.
    conn = duckdb.connect(config_module.settings.duckdb_path, read_only=True)
    try:
        expected_values = {
            row[0].isoformat(): row[1]
            for row in conn.execute(
                "SELECT ts, value FROM events WHERE service = ? AND metric_name = ?",
                [anomaly["service"], anomaly["metric_name"]],
            ).fetchall()
        }
    finally:
        conn.close()
    for point in body["events"]:
        assert point["ts"] in expected_values
        assert point["value"] == expected_values[point["ts"]]

    # At least one point must fall inside the flagged window itself — an
    # unbounded upper edge (M6) would still pass a "some events came back"
    # check but silently drift past the anomaly's own end_ts.
    assert any(point["in_window"] for point in body["events"])
    for point in body["events"]:
        if point["in_window"]:
            assert anomaly["start_ts"] <= point["ts"] <= anomaly["end_ts"]

    # And at least one point must fall OUTSIDE it, on each side — this is
    # the one round 1's fixes added and round 2 found completely untested:
    # zeroing the padding (or shrinking _DRILLDOWN_MIN_PAD to ~nothing)
    # collapses every anomaly back to a 2-point flat line with the full
    # suite green, silently undoing the whole point of the fix (Epic 7
    # review round 2, High #1, N4/N8/N10).
    all_ts = [point["ts"] for point in body["events"]]
    assert any(not point["in_window"] for point in body["events"])
    assert min(all_ts) < anomaly["start_ts"]
    assert max(all_ts) > anomaly["end_ts"]

    # The returned span must actually SCALE with this anomaly's own width,
    # not a fixed pad — calling _drilldown_pad(_DRILLDOWN_MIN_PAD) instead
    # of the real width passed every assertion above (still some
    # out-of-window context, just less) while halving the context on the
    # widest real anomalies (Epic 7 review round 3, High #1, O2). Checked
    # as "pad exceeds the floor", not an exact grid-aligned boundary — the
    # widest anomaly can sit close enough to the corpus edge that an exact
    # `min(ts) == start_ts - width` isn't reliable, but a fixed-floor pad
    # (O2) can never itself exceed the floor, so this still separates them.
    start_dt = datetime.fromisoformat(anomaly["start_ts"])
    end_dt = datetime.fromisoformat(anomaly["end_ts"])
    width = end_dt - start_dt
    min_ts_dt, max_ts_dt = datetime.fromisoformat(min(all_ts)), datetime.fromisoformat(max(all_ts))
    if width > _DRILLDOWN_MIN_PAD:
        assert start_dt - min_ts_dt > _DRILLDOWN_MIN_PAD
        assert max_ts_dt - end_dt > _DRILLDOWN_MIN_PAD

    # The right-hand pad must stay within the documented ceiling — nothing
    # else checks the two sides are independently bounded (Epic 7 review
    # round 3, High #1, O10).
    assert max_ts_dt - end_dt <= _DRILLDOWN_MAX_PAD

    # Response ordering is part of the contract, same as the list endpoint's
    # (Epic 7 review round 3, nit N2).
    assert all_ts == sorted(all_ts)

    # `in_window`'s boundary, independently re-derived per point rather than
    # just "some in-window point exists" — a left-strict `<` on the lower
    # bound passed every assertion above, and on a window with exactly one
    # sample it makes `in_window` false for every point, silently hiding
    # the UI's flagged-region band entirely (Epic 7 review round 4, Low #3,
    # Q4 — the same failure mode round 2's nit N1 fixed, re-entering through
    # the boundary instead of the band's own min/max).
    for point in body["events"]:
        point_dt = datetime.fromisoformat(point["ts"])
        assert point["in_window"] == (start_dt <= point_dt <= end_dt)

    # The two pads must be roughly symmetric — nothing else checks the left
    # side isn't shrunk independently of the right, which "some context on
    # each side" and "right side stays under the ceiling" both miss (Epic 7
    # review round 4, Low #3, Q7). Tolerant of one sampling interval either
    # way (this fixture's generator uses interval_minutes=15).
    left_pad, right_pad = start_dt - min_ts_dt, max_ts_dt - end_dt
    assert left_pad >= right_pad - timedelta(minutes=20)
    assert right_pad >= left_pad - timedelta(minutes=20)


def test_get_anomaly_detail_events_are_capped(tmp_path, monkeypatch):
    # _DRILLDOWN_MAX_PAD bounds the PAD, not the window itself — a wide
    # anomaly (this project injects sustained_drift, which can span most of
    # the corpus) drove an unbounded response independent of the pad cap;
    # verified live a whole-corpus window returned 4,032 events in one
    # response with the pad cap unchanged (Epic 7 review round 3, Low #2).
    monkeypatch.setattr(config_module.settings, "duckdb_path", str(tmp_path / "ops.duckdb"))
    monkeypatch.setattr(config_module.settings, "ground_truth_duckdb_path", str(tmp_path / "gt.duckdb"))
    monkeypatch.setattr(config_module.settings, "query_log_duckdb_path", str(tmp_path / "query_log.duckdb"))
    events_df, gt_df = generate(seed=1, days=30, interval_minutes=5)
    write_to_db(events_df, gt_df)
    run_detector()

    conn = duckdb.connect(config_module.settings.duckdb_path, read_only=False)
    try:
        row = conn.execute("SELECT service, metric_name, MIN(ts), MAX(ts) FROM events GROUP BY service, metric_name LIMIT 1").fetchone()
        service, metric_name, min_ts, max_ts = row
        conn.execute(
            "INSERT INTO detected_anomalies VALUES (100000, ?, ?, ?, ?, 'test', 99.0, [])",
            [service, metric_name, min_ts, max_ts],
        )
    finally:
        conn.close()

    response = client.get("/anomalies/100000")
    assert response.status_code == 200
    assert len(response.json()["events"]) <= 2000


def test_get_anomaly_detail_pad_is_capped_through_the_endpoint(tmp_path, monkeypatch):
    # _drilldown_pad()'s ceiling is correct in isolation (see
    # test_drilldown_pad_floor_and_ceiling) but nothing tied that to the
    # endpoint actually calling it — inlining an uncapped
    # `max(width, _DRILLDOWN_MIN_PAD)` directly in get_anomaly instead of
    # calling the helper passed every other test here (Epic 7 review round
    # 3, High #1, O1). A wide anomaly with plenty of surrounding corpus is
    # the only way to observe the cap: padding is capped at 6h, so an
    # anomaly plenty wider than that must NOT get a proportionally wider
    # window.
    monkeypatch.setattr(config_module.settings, "duckdb_path", str(tmp_path / "ops.duckdb"))
    monkeypatch.setattr(config_module.settings, "ground_truth_duckdb_path", str(tmp_path / "gt.duckdb"))
    monkeypatch.setattr(config_module.settings, "query_log_duckdb_path", str(tmp_path / "query_log.duckdb"))
    events_df, gt_df = generate(seed=1, days=30, interval_minutes=5)
    write_to_db(events_df, gt_df)
    run_detector()

    conn = duckdb.connect(config_module.settings.duckdb_path, read_only=False)
    try:
        service, metric_name, corpus_min, corpus_max = conn.execute(
            "SELECT service, metric_name, MIN(ts), MAX(ts) FROM events GROUP BY service, metric_name LIMIT 1"
        ).fetchone()
        # A 12h window sitting comfortably mid-corpus (plenty of real data
        # on both sides to distinguish a 6h-capped pad from a 12h uncapped
        # one, well clear of edge truncation).
        anomaly_start = corpus_min + (corpus_max - corpus_min) / 2
        anomaly_end = anomaly_start + timedelta(hours=12)
        conn.execute(
            "INSERT INTO detected_anomalies VALUES (100001, ?, ?, ?, ?, 'test', 99.0, [])",
            [service, metric_name, anomaly_start, anomaly_end],
        )
    finally:
        conn.close()

    body = client.get("/anomalies/100001").json()
    all_ts = [datetime.fromisoformat(p["ts"]) for p in body["events"]]
    # Uncapped, pad would equal the 12h width, putting the earliest/latest
    # event ~12h before/after the window. Capped at 6h, they can be at most
    # ~6h out (plus one sampling interval of slack for grid snapping).
    assert min(all_ts) >= anomaly_start - timedelta(hours=6, minutes=10)
    assert max(all_ts) <= anomaly_end + timedelta(hours=6, minutes=10)


def test_get_anomaly_detail_404_for_unknown_id(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    response = client.get("/anomalies/999999")
    assert response.status_code == 404


def test_get_anomaly_detail_422_for_out_of_range_id(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    # DuckDB's Python-int binding falls back to DOUBLE and raises past
    # double-max, which reached the client as a bare 500 before the `Path`
    # bound (Epic 7 review round 1, Medium #2).
    response = client.get(f"/anomalies/{'9' * 309}")
    assert response.status_code == 422
