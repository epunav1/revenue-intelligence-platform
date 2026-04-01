"""
DuckDB-based analytics layer — DBT-style staging → intermediate → mart.

Each SQL file in src/models/{layer}/ is executed in dependency order.
Results are materialised as DuckDB tables AND exported to parquet.
"""
import logging
import re
from pathlib import Path

import duckdb
import pandas as pd

from src.config import DB_PATH, RAW_DIR, STG_DIR, INT_DIR, MART_DIR, MODELS_DIR

log = logging.getLogger(__name__)

# Layer execution order
LAYERS = [
    ("staging",      STG_DIR),
    ("intermediate", INT_DIR),
    ("mart",         MART_DIR),
]


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a persistent DuckDB connection to the platform database."""
    conn = duckdb.connect(str(DB_PATH))
    conn.execute("SET threads TO 4")
    conn.execute("SET memory_limit='2GB'")
    return conn


def _register_raw_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Load raw parquet files as DuckDB views."""
    for parquet in RAW_DIR.glob("*.parquet"):
        table_name = f"raw_{parquet.stem}"
        conn.execute(
            f"CREATE OR REPLACE VIEW {table_name} AS "
            f"SELECT * FROM read_parquet('{parquet}')"
        )
        log.debug("Registered view: %s", table_name)


def _resolve_order(sql_dir: Path) -> list[Path]:
    """
    Return SQL files sorted by their optional leading order number
    (e.g. 01_stg_customers.sql → first).
    Files without a prefix sort lexicographically after numbered ones.
    """
    files = list(sql_dir.glob("*.sql"))
    def _key(p: Path):
        m = re.match(r"^(\d+)_", p.name)
        return (int(m.group(1)) if m else 999, p.name)
    return sorted(files, key=_key)


def run_layer(conn: duckdb.DuckDBPyConnection,
              layer: str,
              output_dir: Path) -> None:
    sql_dir = MODELS_DIR / layer
    if not sql_dir.exists():
        log.warning("Layer directory not found: %s", sql_dir)
        return

    for sql_file in _resolve_order(sql_dir):
        model_name = re.sub(r"^\d+_", "", sql_file.stem)
        log.info("  Running model: %s/%s", layer, model_name)
        sql = sql_file.read_text()
        # replace {{ref('...')}} style with direct table names (lightweight dbt-style)
        sql = re.sub(r"\{\{\s*ref\('([^']+)'\)\s*\}\}", r"\1", sql)
        try:
            conn.execute(f"CREATE OR REPLACE TABLE {model_name} AS\n{sql}")
            # export to parquet
            out = output_dir / f"{model_name}.parquet"
            conn.execute(f"COPY {model_name} TO '{out}' (FORMAT PARQUET)")
            count = conn.execute(f"SELECT COUNT(*) FROM {model_name}").fetchone()[0]
            log.info("    → %d rows materialised → %s", count, out.name)
        except Exception as exc:
            log.error("    FAILED: %s — %s", model_name, exc)
            raise


def build_all(conn: duckdb.DuckDBPyConnection | None = None) -> duckdb.DuckDBPyConnection:
    """Run all layers in order and return the open connection."""
    if conn is None:
        conn = get_connection()
    _register_raw_tables(conn)
    for layer, out_dir in LAYERS:
        log.info("▶ Layer: %s", layer.upper())
        run_layer(conn, layer, out_dir)
    log.info("✓ All models built.")
    return conn


def query(sql: str, conn: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Convenience wrapper — execute SQL and return a DataFrame."""
    if conn is None:
        conn = get_connection()
    return conn.execute(sql).df()


def load_mart(table: str, conn: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Load a mart table, preferring the live DuckDB table over parquet."""
    if conn is None:
        conn = get_connection()
    try:
        return conn.execute(f"SELECT * FROM {table}").df()
    except Exception:
        parquet = MART_DIR / f"{table}.parquet"
        if parquet.exists():
            return pd.read_parquet(parquet)
        raise FileNotFoundError(f"Mart table '{table}' not found in DB or parquet.")
