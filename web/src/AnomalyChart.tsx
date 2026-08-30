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
        {/* var(--x, #hex) below, not bare var(--x) — presentation attributes
            (as opposed to a CSS rule or inline style) historically used SVG
            parse grammar rather than CSS, so var() resolution here is a
            genuinely contested behaviour. Verified it resolves in the
            Chromium this app targets, but a fallback closes the silent-
            invisible-chart failure mode on anything that disagrees — same
            class of risk finding #8 closed for the band (review round 2,
            NEW-4). The fallback is the source of truth only if the token
            fails to resolve; the CSS variable still wins everywhere else. */}
        <CartesianGrid stroke="var(--border, #262b33)" strokeDasharray="3 3" />
        <XAxis
          dataKey="ts"
          type="number"
          domain={["dataMin", "dataMax"]}
          tickFormatter={(t) => new Date(t).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
          stroke="var(--muted, #9aa2ad)"
          tick={{ fontSize: 11 }}
        />
        <YAxis stroke="var(--muted, #9aa2ad)" tick={{ fontSize: 11 }} label={{ value: detail.metric_name, angle: -90, position: "insideLeft", fill: "var(--muted, #9aa2ad)", fontSize: 11 }} />
        <Tooltip
          labelFormatter={(t) => new Date(t as number).toLocaleString()}
          contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} // a React style object -> inline style, never subject to the SVG-presentation-attribute doubt above
        />
        <ReferenceArea
          x1={new Date(detail.start_ts).getTime()}
          x2={new Date(detail.end_ts).getTime()}
          // Recharts' default (ifOverflow="discard") drops the whole band —
          // silently, no error — the moment either edge falls outside the
          // axis domain (["dataMin","dataMax"], i.e. the events actually
          // returned). The endpoint's own event LIMIT (app/main.py) means a
          // wide anomaly can legitimately have its window extend past what
          // was returned; extendDomain stretches the axis to fit instead,
          // which also makes any such truncation visible as empty space
          // rather than hiding the one visual this chart exists to show
          // (review round 1, finding #8 — verified not currently live
          // against the real corpus, but a silent failure mode worth
          // closing for one attribute).
          ifOverflow="extendDomain"
          fill="var(--danger, #ff4b4b)"
          fillOpacity={0.15}
          stroke="none"
        />
        <Line
          type="linear" // not "monotone" — the metric is sampled, not continuous, and a
          // spline invents curvature between points that was never measured,
          // softening exactly the sharp transitions that make a spike
          // legible against the shaded window (review round 1, finding #16;
          // the Altair original this replaces used a plain, unsmoothed line).
          dataKey="value"
          name={detail.metric_name} // otherwise the tooltip's series label is the literal
          // string "value" instead of e.g. "latency_ms" (review round 1, N3).
          stroke="var(--accent, #6ee7b7)"
          dot={data.length <= 200} // a capped response can return up to 2000 points; an
          // unconditional dot on all of them is 2000 SVG circles for no
          // visual gain past a certain density (review round 1, N4).
          strokeWidth={1.5}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
