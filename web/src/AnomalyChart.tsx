import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AnomalyDetail } from "./api";

// Direct port of the old Streamlit UI's Altair band+line chart
// (ui/streamlit_app.py) — ReferenceArea is Recharts' equivalent of Altair's
// mark_rect band, shaded over the anomaly's own start_ts/end_ts (not the
// min/max of in-window points, which understates a narrow anomaly and goes
// zero-width when only one sample falls inside the window — same reasoning
// as the original).
export function AnomalyChart({ detail }: { detail: AnomalyDetail }) {
  const data = detail.events.map((e) => ({ ts: new Date(e.ts).getTime(), value: e.value }));

  if (data.length === 0) {
    return <p className="empty-state">No raw events recorded in this window.</p>;
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid stroke="#262b33" strokeDasharray="3 3" />
        <XAxis
          dataKey="ts"
          type="number"
          domain={["dataMin", "dataMax"]}
          tickFormatter={(t) => new Date(t).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
          stroke="#9aa2ad"
          tick={{ fontSize: 11 }}
        />
        <YAxis stroke="#9aa2ad" tick={{ fontSize: 11 }} label={{ value: detail.metric_name, angle: -90, position: "insideLeft", fill: "#9aa2ad", fontSize: 11 }} />
        <Tooltip
          labelFormatter={(t) => new Date(t as number).toLocaleString()}
          contentStyle={{ background: "#14171c", border: "1px solid #262b33", borderRadius: 8, fontSize: 12 }}
        />
        <ReferenceArea
          x1={new Date(detail.start_ts).getTime()}
          x2={new Date(detail.end_ts).getTime()}
          fill="#ff4b4b"
          fillOpacity={0.15}
          stroke="none"
        />
        <Line type="monotone" dataKey="value" stroke="#6ee7b7" dot={{ r: 2 }} strokeWidth={1.5} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
