import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

# On Streamlit Community Cloud, API_URL MUST be set in the app's Secrets —
# the localhost fallback only makes sense for local dev (same pattern as
# Project 3's UI).
API_URL = os.environ.get("API_URL", "http://localhost:8000")

_MAX_CHAT_TURNS = 50
# Matches the API's own _ANOMALIES_MAX_LIMIT — sent explicitly on every
# browse-tab request so a corpus that ever grows past the endpoint's
# default (200) truncates loudly (via the table's own row count) rather
# than the UI silently never asking for the rest (Epic 7 review round 2,
# nit N5).
_BROWSE_LIMIT = 500


def _api_error_text(e: requests.RequestException) -> str:
    # str(e) on a real deploy renders the backend's own URL to the public UI
    # (e.g. "404 Client Error: Not Found for url: https://internal-api...")
    # (Epic 7 review round 1, nit N3). A response we got at all just means a
    # non-2xx status; anything else (DNS, timeout, connection refused) never
    # reached a server worth naming.
    resp = getattr(e, "response", None)
    if resp is not None:
        return f"API returned {resp.status_code}"
    return "could not reach the API"


@st.cache_data(ttl=60)
def _get_anomalies(service: str, metric: str) -> list[dict]:
    params = {"limit": _BROWSE_LIMIT}
    if service:
        params["service"] = service
    if metric:
        params["metric"] = metric
    resp = requests.get(f"{API_URL}/anomalies", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=60)
def _get_anomaly_detail(anomaly_id: int) -> dict:
    resp = requests.get(f"{API_URL}/anomalies/{anomaly_id}", timeout=15)
    resp.raise_for_status()
    return resp.json()


st.set_page_config(page_title="AI Ops Observability Copilot", page_icon="📈")
st.title("📈 AI Ops Observability Copilot")
st.caption(
    "Ask questions about synthetic service logs/metrics in plain English, "
    "or browse anomalies a classical stats detector flagged and drill into "
    "the raw data behind each one."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {question, answer, anomaly_ids, error}
if "drilldown_id" not in st.session_state:
    st.session_state.drilldown_id = None

tab_chat, tab_browse = st.tabs(["💬 Ask", "🔍 Browse anomalies"])

with tab_chat:
    for i, turn in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            if turn["error"]:
                st.error(turn["error"])
            else:
                st.write(turn["answer"])
                for aid in turn["anomaly_ids"]:
                    # Index into chat_history, not id(turn) — a dict's id()
                    # only happens to stay unique/stable today because every
                    # turn is kept alive simultaneously; indexing is
                    # unique-by-construction and doesn't depend on that
                    # holding (Epic 7 review round 1, nit N2).
                    if st.button(f"View flagged anomaly #{aid} →", key=f"link-{i}-{aid}"):
                        st.session_state.drilldown_id = aid
                        st.rerun()

    question = st.chat_input("e.g. Was there a latency spike on checkout-api yesterday?")
    if question:
        st.session_state.chat_history.append({"question": question, "answer": None, "anomaly_ids": [], "error": None})
        st.session_state.chat_history = st.session_state.chat_history[-_MAX_CHAT_TURNS:]
        try:
            resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            st.session_state.chat_history[-1].update(
                answer=body["answer"], anomaly_ids=body["anomaly_ids"], error=body["error"]
            )
        except requests.RequestException as e:
            st.session_state.chat_history[-1]["error"] = _api_error_text(e)
        st.rerun()

with tab_browse:
    col1, col2 = st.columns(2)
    service_filter = col1.text_input("Filter by service", key="service_filter")
    metric_filter = col2.text_input("Filter by metric", key="metric_filter")

    # None means "couldn't ask", [] means "asked, no matches" — collapsing
    # these into one falsy value made a dead API render "No anomalies match
    # this filter.", which reads as an invitation to loosen filters instead
    # of the actual problem (Epic 7 review round 1, Low #5).
    anomalies = None
    try:
        anomalies = _get_anomalies(service_filter, metric_filter)
    except requests.RequestException as e:
        st.error(f"Could not load anomalies: {_api_error_text(e)}")

    if anomalies:
        # `limit=_BROWSE_LIMIT` raised the silent-truncation threshold from
        # the endpoint's default (200) to 500, but 500 is a hard ceiling the
        # UI can't page past — this project's own history shows the row
        # count is a tuning parameter (a bare 3-sigma threshold once
        # produced 595), so a truncation needs to be visible, not just less
        # likely (Epic 7 review round 3, Low #3).
        caption = f"{len(anomalies)} anomalies"
        if len(anomalies) == _BROWSE_LIMIT:
            caption += f" (showing the newest {_BROWSE_LIMIT})"
        st.caption(caption)
        table = pd.DataFrame(anomalies)[["id", "service", "metric_name", "start_ts", "end_ts", "method", "score"]]
        st.dataframe(table, use_container_width=True, hide_index=True)
        chosen_id = st.selectbox(
            "Drill into anomaly id", options=[a["id"] for a in anomalies], key="browse_select"
        )
        if st.button("Drill down", key="browse_drilldown"):
            st.session_state.drilldown_id = chosen_id
            st.rerun()
    elif anomalies is not None:
        st.info("No anomalies match this filter.")

if st.session_state.drilldown_id is not None:
    st.divider()
    st.subheader(f"Anomaly #{st.session_state.drilldown_id}")
    try:
        detail = _get_anomaly_detail(st.session_state.drilldown_id)
        st.write(
            f"**{detail['service']} / {detail['metric_name']}** — "
            f"{detail['method']} flagged this window (score {detail['score']:.2f}), "
            f"{detail['start_ts']} → {detail['end_ts']}"
        )
        events_df = pd.DataFrame(detail["events"])
        if not events_df.empty:
            events_df["ts"] = pd.to_datetime(events_df["ts"])
            # Shade the flagged window inside its surrounding context —
            # the API now pads the query with baseline data either side,
            # but a bare line chart still can't show which part is the
            # anomaly (Epic 7 review round 1, Medium #3).
            # .interactive() goes on the line layer BEFORE layering, not on
            # the combined chart — bound to the combined chart it attaches
            # to whichever view Altair picks (the rect band, which has no y
            # encoding at all), giving pan/zoom that silently misbehaves
            # depending on the data (Epic 7 review round 2, nit N2).
            line = alt.Chart(events_df).mark_line(point=True).encode(
                x=alt.X("ts:T", title="time"), y=alt.Y("value:Q", title=detail["metric_name"]),
            ).interactive()
            # Use the anomaly's own start_ts/end_ts (already in `detail`),
            # not the min/max of the sampled in-window points — the latter
            # understates the flagged region by up to one sample interval
            # and goes zero-width (invisible) when only one point falls
            # inside the window (Epic 7 review round 2, nit N1).
            # start_ts/end_ts are always present in `detail` regardless of
            # whether any sampled point landed inside the window (a narrow
            # off-grid anomaly can have zero in-window samples) — gating the
            # band on `in_window.any()` made the flagged region invisible on
            # exactly those windows even though the API told the UI exactly
            # where it is (Epic 7 review round 4, nit N2).
            band = alt.Chart(pd.DataFrame({
                "start": [pd.Timestamp(detail["start_ts"])], "end": [pd.Timestamp(detail["end_ts"])],
            })).mark_rect(color="#ff4b4b", opacity=0.15).encode(x="start:T", x2="end:T")
            st.altair_chart(band + line, use_container_width=True)
        else:
            st.info("No raw events recorded in this window.")
    except requests.RequestException as e:
        st.error(f"Could not load anomaly detail: {_api_error_text(e)}")
    if st.button("Close drill-down"):
        st.session_state.drilldown_id = None
        st.rerun()
