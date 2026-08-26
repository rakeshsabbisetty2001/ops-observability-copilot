"""DuckDB connection helper.

Two explicit modes, never a default that could accidentally allow writes:
- read_only=True  -> the API's request-time path (text-to-SQL, /anomalies).
- read_only=False -> offline scripts only (data generation, detector runs).
"""
import duckdb

from app.config import settings


def get_connection(read_only: bool) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(settings.duckdb_path, read_only=read_only)
