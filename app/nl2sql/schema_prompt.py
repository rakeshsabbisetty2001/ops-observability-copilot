"""Fixed schema-grounding text for text-to-SQL generation.

Deliberately static, never built from a live DESCRIBE/SHOW query — a live
introspection would let the question itself influence what schema text the
model sees, which is one more thing an adversarial question could try to
manipulate. This text is the same for every request.
"""

SCHEMA_DESCRIPTION = """You can query exactly two tables in a DuckDB database:

events
  id            BIGINT      unique row id
  ts            TIMESTAMP   when this log/metric point was recorded
  service       VARCHAR     e.g. 'checkout-api', 'payments-worker', 'auth-service'
  metric_name   VARCHAR     e.g. 'latency_ms', 'error_rate', 'cpu_pct'
  value         DOUBLE      the metric's value at this timestamp
  level         VARCHAR     log severity: 'info', 'warn', or 'error'
  message       VARCHAR     a short human-readable log line

detected_anomalies
  id                BIGINT      unique row id
  service           VARCHAR
  metric_name       VARCHAR
  start_ts          TIMESTAMP   when the flagged window began
  end_ts            TIMESTAMP   when the flagged window ended
  method            VARCHAR     which detector(s) flagged it, e.g. 'seasonal_residual'
  score             DOUBLE      detector's z-score for this window (not comparable across methods)
  sample_event_ids  BIGINT[]    a few representative events from the window

No other tables exist and none may be referenced. Write exactly one SQL
SELECT statement — no semicolons, no comments, no other statement types, no
markdown formatting. Output only the SQL text."""
