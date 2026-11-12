"""Medallion lakehouse on Delta Lake: bronze (raw) -> silver (clean) -> gold (KPIs).
Same format Microsoft Fabric OneLake uses.
Run:  python -m src.lakehouse            (build all)
      python -m src.lakehouse --scrape 10 (scrape 10 containers first, then build)"""
import json
import sys
from datetime import datetime, timezone

import duckdb
import pandas as pd
from deltalake import DeltaTable, write_deltalake

from src import config

DELAY_THRESHOLD_HOURS = 48  # rule from data/knowledge/shipment_status_codes.md


def _path(layer: str, name: str) -> str:
    return str(config.LAKEHOUSE_DIR / layer / name)


def exists(layer: str, name: str) -> bool:
    return DeltaTable.is_deltatable(_path(layer, name))


def read(layer: str, name: str) -> pd.DataFrame:
    return DeltaTable(_path(layer, name)).to_pandas()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============================== BRONZE (raw, never edited) ==============================
def ingest_erp_csv() -> int:
    df = pd.read_csv(config.RAW_DIR / "erp_shipments.csv", dtype=str)  # all strings: keep raw values exactly
    df["_ingested_at"] = _now()          # lineage: when did we load it?
    df["_source"] = "erp_shipments.csv"  # lineage: where did it come from?
    write_deltalake(_path("bronze", "erp_shipments"), df, mode="overwrite")  # full snapshot each run
    return len(df)


def ingest_tracking(results: list[dict]) -> int:
    rows = [{"container_no": r["container_no"], "payload": json.dumps(r), "_ingested_at": _now()} for r in results]
    if rows:  # append = keep history of every scrape
        write_deltalake(_path("bronze", "portal_tracking"), pd.DataFrame(rows), mode="append")
    return len(rows)


# ============================== SILVER (clean + merged) ==============================
def _parse_dates(s: pd.Series) -> pd.Series:
    """Try the 3 formats found in Step 4; keep the first one that works per value."""
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    for fmt in ("%d-%b-%Y", "%d/%m/%Y"):  # DD/MM (Sri Lankan convention), never MM/DD
        out = out.fillna(pd.to_datetime(s, format=fmt, errors="coerce"))
    return out


def _latest_tracking() -> pd.DataFrame:
    """Latest successful portal scrape per container -> live_eta, live_status."""
    if not exists("bronze", "portal_tracking"):
        return pd.DataFrame(columns=["container_no", "live_eta", "live_status"])
    payloads = [json.loads(p) for p in read("bronze", "portal_tracking")["payload"]]
    t = pd.DataFrame([p for p in payloads if "error" not in p])  # skip not_found / timeout rows
    if t.empty:
        return pd.DataFrame(columns=["container_no", "live_eta", "live_status"])
    t = t.sort_values("scraped_at").drop_duplicates("container_no", keep="last")
    return pd.DataFrame({
        "container_no": t["container_no"],
        "live_eta": pd.to_datetime(t["live_eta"]),
        "live_status": t["latest_status"],
    })


def build_silver() -> int:
    df = read("bronze", "erp_shipments")
    df = df.drop_duplicates("shipment_id").copy()                       # rule 1: duplicates
    df["container_no"] = df["container_no"].str.strip().str.upper()    # rule 2: lowercase containers
    for col in ("etd", "original_eta", "current_eta"):                 # rule 3: 3 date formats
        df[col] = _parse_dates(df[col])
    df["teu"] = pd.to_numeric(df["teu"], errors="coerce").fillna(1).astype(int)  # rule 4: empty teu
    df["freight_usd"] = pd.to_numeric(df["freight_usd"], errors="coerce")

    df = df.merge(_latest_tracking(), on="container_no", how="left")   # add live portal data
    df["live_eta"] = pd.to_datetime(df["live_eta"])                    # stays a date type even if no scrapes yet
    best_eta = df["live_eta"].fillna(df["current_eta"])                # portal knows best, else ERP
    df["delay_hours"] = (best_eta - df["original_eta"]).dt.total_seconds() / 3600
    df["is_delayed"] = df["delay_hours"] > DELAY_THRESHOLD_HOURS

    df = df.drop(columns=["_ingested_at", "_source"])
    for col in ("etd", "original_eta", "current_eta", "live_eta"):
        df[col] = df[col].astype("datetime64[us]")  # Delta stores microseconds
    write_deltalake(_path("silver", "shipments"), df, mode="overwrite", schema_mode="overwrite")
    return len(df)


# ============================== GOLD (business KPIs, SQL) ==============================
def build_gold() -> None:
    shipments = read("silver", "shipments")  # DuckDB can query this pandas variable by name
    by_carrier = duckdb.sql("""
        SELECT carrier,
               COUNT(*)                                                      AS shipments,
               CAST(SUM(teu) AS INT)                                         AS teu,
               ROUND(100.0 * AVG(CASE WHEN is_delayed THEN 0 ELSE 1 END), 1) AS on_time_pct,
               ROUND(AVG(delay_hours), 1)                                    AS avg_delay_hours
        FROM shipments GROUP BY carrier ORDER BY on_time_pct
    """).df()
    by_port = duckdb.sql("""
        SELECT pod                          AS port,
               COUNT(*)                     AS shipments,
               CAST(SUM(CAST(is_delayed AS INT)) AS INT) AS delayed,
               CAST(SUM(freight_usd) AS BIGINT)          AS freight_usd
        FROM shipments GROUP BY pod ORDER BY delayed DESC
    """).df()
    write_deltalake(_path("gold", "kpi_by_carrier"), by_carrier, mode="overwrite", schema_mode="overwrite")
    write_deltalake(_path("gold", "kpi_by_port"), by_port, mode="overwrite", schema_mode="overwrite")


def query(sql: str) -> pd.DataFrame:
    """SQL over all silver/gold tables, named silver_<name> / gold_<name> (like Fabric's SQL endpoint)."""
    con = duckdb.connect()
    for layer in ("silver", "gold"):
        folder = config.LAKEHOUSE_DIR / layer
        for table in (folder.iterdir() if folder.exists() else []):
            con.register(f"{layer}_{table.name}", read(layer, table.name))
    return con.sql(sql).df()


def build_all() -> None:
    print("bronze erp rows :", ingest_erp_csv())
    print("silver rows     :", build_silver())
    build_gold()
    print("gold tables built")


if __name__ == "__main__":
    if "--scrape" in sys.argv:  # optional: pull live ETAs from the portal first
        from src.scraper import track_containers_sync
        n = int(sys.argv[sys.argv.index("--scrape") + 1])
        ingest_erp_csv()  # make sure bronze exists so we know which containers to scrape
        containers = read("bronze", "erp_shipments")["container_no"].str.strip().str.upper().unique()[:n].tolist()
        print("scraped         :", ingest_tracking(track_containers_sync(containers)))
    build_all()
    print(query("SELECT * FROM gold_kpi_by_carrier").to_string(index=False))
