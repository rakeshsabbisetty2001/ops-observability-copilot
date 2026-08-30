import { useEffect, useMemo, useRef, useState } from "react";
import { AnomalyChart } from "./AnomalyChart";
import {
  ask,
  getAnomalies,
  getAnomalyDetail,
  BROWSE_LIMIT,
  type AnomalyDetail,
  type AnomalySummary,
} from "./api";

interface ChatTurn {
  question: string;
  answer: string | null;
  anomalyIds: number[];
  error: string | null;
}

const MAX_CHAT_TURNS = 50; // ported from ui/streamlit_app.py

type SortKey = "start_ts" | "score" | "service" | "metric_name";

export default function App() {
  const [tab, setTab] = useState<"ask" | "browse">("ask");
  const [drilldownId, setDrilldownId] = useState<number | null>(null);
  const [allAnomalies, setAllAnomalies] = useState<AnomalySummary[] | null>(null);

  // Fetched once (unfiltered) to drive the stat strip and give Browse its
  // service/metric universe — the filtered fetch in BrowseTab is separate,
  // this is just for the header numbers.
  useEffect(() => {
    getAnomalies("", "")
      .then(setAllAnomalies)
      .catch(() => setAllAnomalies([]));
  }, []);

  const stats = useMemo(() => {
    if (!allAnomalies) return null;
    return {
      total: allAnomalies.length,
      services: new Set(allAnomalies.map((a) => a.service)).size,
    };
  }, [allAnomalies]);

  return (
    <>
      <header className="app-header">
        <h1>📈 AI Ops Observability Copilot</h1>
        <p>
          Ask questions about synthetic service logs/metrics in plain English, or browse anomalies a
          classical stats detector flagged and drill into the raw data behind each one.
        </p>
      </header>

      <div className="stat-strip">
        <StatTile label="anomalies flagged" value={stats?.total} />
        <StatTile label="services monitored" value={stats?.services} />
      </div>

      <div className="tabs">
        <button className={tab === "ask" ? "active" : ""} onClick={() => setTab("ask")}>
          💬 Ask
        </button>
        <button className={tab === "browse" ? "active" : ""} onClick={() => setTab("browse")}>
          🔍 Browse anomalies
        </button>
      </div>

      {tab === "ask" && <AskTab onViewAnomaly={setDrilldownId} />}
      {tab === "browse" && <BrowseTab onDrilldown={setDrilldownId} />}

      {drilldownId !== null && <Drilldown id={drilldownId} onClose={() => setDrilldownId(null)} />}
    </>
  );
}

function StatTile({ label, value }: { label: string; value: number | undefined }) {
  // Simple count-up on mount, skipped entirely under prefers-reduced-motion —
  // same pattern as the portfolio site's own animated stats, whose review
  // chain flagged an un-gated version as the most-liked but also most
  // motion-sensitive part of that page.
  const [display, setDisplay] = useState(0);
  const reduceMotion = useRef(
    typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );

  useEffect(() => {
    if (value === undefined) return;
    if (reduceMotion.current) {
      setDisplay(value);
      return;
    }
    const start = performance.now();
    const duration = 500;
    let raf: number;
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      setDisplay(Math.round(value * t));
      if (t < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [value]);

  return (
    <div className="stat-tile">
      <div className="value">{value === undefined ? "–" : display}</div>
      <div className="label">{label}</div>
    </div>
  );
}

function AskTab({ onViewAnomaly }: { onViewAnomaly: (id: number) => void }) {
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit() {
    const q = question.trim();
    if (!q || loading) return;
    setQuestion("");
    setLoading(true);
    const turn: ChatTurn = { question: q, answer: null, anomalyIds: [], error: null };
    setHistory((h) => [...h, turn].slice(-MAX_CHAT_TURNS));
    try {
      const body = await ask(q);
      setHistory((h) => {
        const next = [...h];
        next[next.length - 1] = { question: q, answer: body.answer, anomalyIds: body.anomaly_ids, error: body.error };
        return next;
      });
    } catch (e) {
      setHistory((h) => {
        const next = [...h];
        next[next.length - 1] = { ...turn, error: (e as Error).message };
        return next;
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="chat-history">
        {history.map((turn, i) => (
          <div key={i}>
            <div className="chat-turn user">{turn.question}</div>
            {turn.error ? (
              <div className="chat-turn assistant error">{turn.error}</div>
            ) : turn.answer !== null ? (
              <div className="chat-turn assistant">
                {turn.answer}
                <div>
                  {turn.anomalyIds.map((aid) => (
                    <button key={aid} className="anomaly-chip" onClick={() => onViewAnomaly(aid)}>
                      View flagged anomaly #{aid} →
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="chat-turn assistant">Thinking…</div>
            )}
          </div>
        ))}
      </div>
      <div className="chat-input-row">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="e.g. Was there a latency spike on checkout-api yesterday?"
          disabled={loading}
        />
        <button onClick={submit} disabled={loading || !question.trim()}>
          Ask
        </button>
      </div>
    </div>
  );
}

function BrowseTab({ onDrilldown }: { onDrilldown: (id: number) => void }) {
  const [service, setService] = useState("");
  const [metric, setMetric] = useState("");
  const [anomalies, setAnomalies] = useState<AnomalySummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("start_ts");
  const [sortDesc, setSortDesc] = useState(true);

  // Debounced live filtering — a real interactivity gain over the old
  // Streamlit UI, which only refetched on Streamlit's own full-script rerun.
  useEffect(() => {
    const t = setTimeout(() => {
      setError(null);
      getAnomalies(service, metric)
        .then(setAnomalies)
        .catch((e) => setError((e as Error).message));
    }, 300);
    return () => clearTimeout(t);
  }, [service, metric]);

  const sorted = useMemo(() => {
    if (!anomalies) return null;
    const copy = [...anomalies];
    copy.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      const cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return sortDesc ? -cmp : cmp;
    });
    return copy;
  }, [anomalies, sortKey, sortDesc]);

  function toggleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDesc((d) => !d);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  }

  return (
    <div>
      <div className="filter-row">
        <input placeholder="Filter by service" value={service} onChange={(e) => setService(e.target.value)} />
        <input placeholder="Filter by metric" value={metric} onChange={(e) => setMetric(e.target.value)} />
      </div>

      {error && <div className="error-banner">Could not load anomalies: {error}</div>}

      {sorted && sorted.length > 0 && (
        <>
          <p className="caption">
            {sorted.length} anomalies{sorted.length === BROWSE_LIMIT ? ` (showing the newest ${BROWSE_LIMIT})` : ""}
            {" — click a row to drill in"}
          </p>
          <table className="anomaly-table">
            <thead>
              <tr>
                <SortHeader label="service" active={sortKey === "service"} desc={sortDesc} onClick={() => toggleSort("service")} />
                <SortHeader label="metric" active={sortKey === "metric_name"} desc={sortDesc} onClick={() => toggleSort("metric_name")} />
                <SortHeader label="start" active={sortKey === "start_ts"} desc={sortDesc} onClick={() => toggleSort("start_ts")} />
                <th>method</th>
                <SortHeader label="score" active={sortKey === "score"} desc={sortDesc} onClick={() => toggleSort("score")} />
              </tr>
            </thead>
            <tbody>
              {sorted.map((a) => (
                <tr key={a.id} className="row-clickable" onClick={() => onDrilldown(a.id)}>
                  <td>{a.service}</td>
                  <td>{a.metric_name}</td>
                  <td>{new Date(a.start_ts).toLocaleString()}</td>
                  <td>{a.method}</td>
                  <td>{a.score.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      {sorted && sorted.length === 0 && <p className="empty-state">No anomalies match this filter.</p>}
    </div>
  );
}

function SortHeader({ label, active, desc, onClick }: { label: string; active: boolean; desc: boolean; onClick: () => void }) {
  return (
    <th onClick={onClick}>
      {label}
      {active ? (desc ? " ↓" : " ↑") : ""}
    </th>
  );
}

function Drilldown({ id, onClose }: { id: number; onClose: () => void }) {
  const [detail, setDetail] = useState<AnomalyDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDetail(null);
    setError(null);
    getAnomalyDetail(id)
      .then(setDetail)
      .catch((e) => setError((e as Error).message));
  }, [id]);

  return (
    <div className="drilldown">
      <div className="drilldown-header">
        <h2 style={{ fontSize: "1.1rem", margin: 0 }}>Anomaly #{id}</h2>
        <button className="drilldown-close" onClick={onClose}>
          Close
        </button>
      </div>
      {error && <div className="error-banner">Could not load anomaly detail: {error}</div>}
      {detail && (
        <>
          <p className="caption">
            <strong>
              {detail.service} / {detail.metric_name}
            </strong>{" "}
            — {detail.method} flagged this window (score {detail.score.toFixed(2)}),{" "}
            {new Date(detail.start_ts).toLocaleString()} → {new Date(detail.end_ts).toLocaleString()}
          </p>
          <AnomalyChart detail={detail} />
        </>
      )}
    </div>
  );
}
