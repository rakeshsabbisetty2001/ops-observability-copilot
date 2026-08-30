// Mirrors app/main.py's response models (AskResponse, AnomalySummary,
// AnomalyDetail) — kept as plain types, not a codegen step, since there are
// only 3 endpoints and the backend contract is frozen (Epic 8).

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface AskResponse {
  question: string;
  sql: string | null;
  answer: string | null;
  row_count: number | null;
  anomaly_ids: number[];
  error: string | null;
}

export interface AnomalySummary {
  id: number;
  service: string;
  metric_name: string;
  start_ts: string;
  end_ts: string;
  method: string;
  score: number;
}

export interface EventPoint {
  ts: string;
  value: number;
  in_window: boolean;
}

export interface AnomalyDetail extends AnomalySummary {
  events: EventPoint[];
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(`${API_URL}${path}`, init);
  } catch {
    // Same three-way error split as the old Streamlit UI's _api_error_text —
    // a response we got at all just means non-2xx; anything else (DNS,
    // timeout, connection refused, cold-start) never reached a server worth
    // naming, and the backend's own URL never gets rendered to the client.
    throw new Error("could not reach the API");
  }
  if (!resp.ok) {
    throw new Error(`API returned ${resp.status}`);
  }
  return resp.json() as Promise<T>;
}

export function ask(question: string): Promise<AskResponse> {
  return apiFetch<AskResponse>("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}

// Matches the API's own _ANOMALIES_MAX_LIMIT (app/main.py) — sent explicitly
// so a corpus that grows past the endpoint's default (200) truncates loudly
// (via the returned row count) instead of the UI silently never asking for
// the rest (ported from the old Streamlit UI's same reasoning).
const BROWSE_LIMIT = 500;

export function getAnomalies(service: string, metric: string): Promise<AnomalySummary[]> {
  const params = new URLSearchParams({ limit: String(BROWSE_LIMIT) });
  if (service) params.set("service", service);
  if (metric) params.set("metric", metric);
  return apiFetch<AnomalySummary[]>(`/anomalies?${params}`);
}

export function getAnomalyDetail(id: number): Promise<AnomalyDetail> {
  return apiFetch<AnomalyDetail>(`/anomalies/${id}`);
}

export { BROWSE_LIMIT };
