#!/usr/bin/env python3
"""
log_analysis.py – robust ETL + analysis for CS485 MPI + CUDA project logs     v0.6.2
───────────────────────────────────────────────────────────────────────────────
▪ v0.6.2:  export() robustness  → works with DuckDB ≥0.10 where
           SHOW TABLES / SHOW VIEWS return (schema,name) tuples.
▪ v0.6.1:  Typer-3 fixes, derived-view rewrite, Agg backend, etc.
"""
from __future__ import annotations
import hashlib, math, os, re, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import duckdb as ddb
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from rich.console import Console
from typer import Typer, Argument, Option

try:
    import magic                         # optional MIME detection
except ModuleNotFoundError:
    magic = None

APP, CON = Typer(add_completion=False), Console()

# ───────────────────────────── constants ──────────────────────────────
WAREHOUSE = Path(".warehouse/cluster_logs.duckdb")
WAREHOUSE.parent.mkdir(exist_ok=True)
HASH_CHUNK, PLOTS_DIR = 1 << 16, Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# ───────────────────────────── utilities ──────────────────────────────
def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        while blk := fh.read(HASH_CHUNK):
            h.update(blk)
    return h.hexdigest()


def _normalise_summary(df: pd.DataFrame, rel: str) -> pd.DataFrame:
    """
    Map any recognised summary-CSV flavour → canonical schema.

    Returns empty DF if schema isn’t recognised.
    """
    df = df.copy()

    # legacy schema
    if {"Timestamp", "Version", "NP", "Time_ms"} <= set(df.columns):
        df["ts"]           = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
        df["version"]      = df["Version"]
        df["np"]           = pd.to_numeric(df["NP"], errors="coerce")
        df["total_time_s"] = pd.to_numeric(df["Time_ms"], errors="coerce") / 1000.0

    # new schema (single-host or multi-host)
    elif {"EntryTimestamp", "ProjectVariant", "NumProcesses"} <= set(df.columns):
        df["ts"]           = pd.to_datetime(df["EntryTimestamp"], utc=True, errors="coerce")
        df["version"]      = df["ProjectVariant"]
        df["np"]           = pd.to_numeric(df["NumProcesses"], errors="coerce")
        col                = "ExecutionTime_ms" if "ExecutionTime_ms" in df.columns else "Time_ms"
        df["total_time_s"] = pd.to_numeric(df[col], errors="coerce") / 1000.0

    else:
        CON.print(f"[yellow]⚠ Unknown summary schema → {rel} skipped[/]")
        return pd.DataFrame(columns=["ts", "version", "np", "total_time_s"])

    return df[["ts", "version", "np", "total_time_s"]]

# ───────────────────────────── ingest ─────────────────────────────────
@APP.command()
def ingest(
    root: Path = Option(..., help="Project root directory"),
    rebuild: bool = Option(False, "--rebuild", help="Drop & rebuild warehouse"),
):
    """Scan *root* → ETL into DuckDB warehouse."""
    if rebuild and WAREHOUSE.exists():
        WAREHOUSE.unlink()
        CON.print("[red]• wiped existing warehouse[/]")

    con = ddb.connect(str(WAREHOUSE))

    # tables
    con.execute(
        """CREATE TABLE IF NOT EXISTS file_index(
             file_id TEXT PRIMARY KEY, relpath TEXT, sha1 TEXT,
             size BIGINT, mtime TIMESTAMP, mime TEXT
        )"""
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS summary_runs(ts TIMESTAMP,version TEXT,np INT,total_time_s DOUBLE)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS run_logs(relpath TEXT,ts TIMESTAMP,version TEXT,np INT,total_time_s DOUBLE)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS source_stats(relpath TEXT,loc INT,func_cnt INT,include_cnt INT,cuda_kernel_cnt INT)"
    )

    seen = {r[0]: r[1] for r in con.execute("SELECT relpath,sha1 FROM file_index").fetchall()}
    tz_utc = timezone.utc

    rows_summary, rows_runlog, rows_srcstats = [], [], []

    for p in root.rglob("*"):
        if p.is_dir():
            continue
        rel = str(p.relative_to(root))
        sha1 = _sha1(p)
        if seen.get(rel) == sha1:
            continue

        size  = p.stat().st_size
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz_utc)
        mime  = magic.from_file(str(p), mime=True) if magic else "application/octet-stream"

        # -------- summary CSV
        if p.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(p)
                ndf = _normalise_summary(df, rel)
                if not ndf.empty:
                    rows_summary.append(ndf)
            except Exception as e:
                CON.print(f"[yellow]⚠ CSV parse failed {rel}: {e}[/]")

        # -------- run log (heuristic)
        elif p.suffix.lower() == ".log" and "run_" in p.name:
            txt = p.read_text(errors="ignore")
            m   = re.search(r"(?:Time|ExecutionTime)_ms[=:]\s*([\d.]+)", txt)
            if m:
                sec  = float(m.group(1)) / 1000.0
                v_m  = re.search(r"v\d(?:_\d\.\d[_\w]+)?", rel)
                np_m = re.search(r"np(\d+)", rel)
                rows_runlog.append(
                    (rel, mtime, v_m.group(0) if v_m else None, int(np_m.group(1)) if np_m else None, sec)
                )

        # -------- source stats
        elif p.suffix.lower() in {".cpp", ".cu", ".hpp"}:
            code = p.read_text(errors="ignore")
            rows_srcstats.append(
                (
                    rel,
                    code.count("\n"),
                    len(re.findall(r"\b[A-Za-z_]\w*\s*\([^)]*\)\s*\{", code)),
                    code.count("#include"),
                    code.count("__global__"),
                )
            )

        # -------- file index
        con.execute(
            "INSERT OR REPLACE INTO file_index VALUES (?,?,?,?,?,?)",
            [sha1, rel, sha1, size, mtime, mime],
        )
        seen[rel] = sha1
        CON.print(f"[green]indexed[/] {rel}")

    # bulk inserts
    if rows_summary:
        df_all = pd.concat(rows_summary, ignore_index=True)
        con.register("df_all", df_all)
        con.execute("INSERT INTO summary_runs SELECT * FROM df_all")
        con.unregister("df_all")
    if rows_runlog:
        con.executemany("INSERT INTO run_logs VALUES (?,?,?,?,?)", rows_runlog)
    if rows_srcstats:
        con.executemany("INSERT INTO source_stats VALUES (?,?,?,?,?)", rows_srcstats)

    # derived views
    con.execute(
        """
        CREATE OR REPLACE VIEW perf_runs AS
        SELECT * FROM (
            SELECT ts,version,np,total_time_s FROM summary_runs
            UNION ALL
            SELECT ts,version,np,total_time_s FROM run_logs
        ) WHERE total_time_s IS NOT NULL;

        CREATE OR REPLACE VIEW best_runs AS
        SELECT DISTINCT ON (version,np) ts,version,np,total_time_s
        FROM perf_runs
        ORDER BY version,np,total_time_s ASC;

        CREATE OR REPLACE VIEW run_stats AS
        SELECT version,np,
               COUNT(*) n,
               AVG(total_time_s)      AS mean_s,
               STDDEV_SAMP(total_time_s) AS sd_s,
               1.96*STDDEV_SAMP(total_time_s)/SQRT(COUNT(*)) AS ci95_s
        FROM perf_runs GROUP BY 1,2;
    """
    )
    con.close()
    CON.print("[bold green]✓ ingest complete[/]")

# ───────────────────────────── quick helpers ─────────────────────────
@APP.command()
def query(sql: str = Argument(..., help="SQL to run")):
    con = ddb.connect(str(WAREHOUSE))
    CON.print(con.execute(sql).fetchdf())
    con.close()

@APP.command()
def stats():   query("SELECT * FROM run_stats ORDER BY version,np;")

@APP.command()
def speedup():
    sql = """
    WITH best AS (
      SELECT version,np,MIN(total_time_s) best FROM perf_runs GROUP BY 1,2
    ),
    base AS (SELECT MIN(best) t1 FROM best WHERE version='V1 Serial' AND np=1)
    SELECT b.version,b.np,b.best, base.t1/b.best AS S,(base.t1/b.best)/b.np AS E
    FROM best b, base ORDER BY S DESC;
    """
    query(sql)

# ───────────────────────────── plots ─────────────────────────────────
@APP.command()
def plot(
    plots: List[str] = Argument(..., help="speedup efficiency")
):
    con = ddb.connect(str(WAREHOUSE))
    df_best = con.execute(
        "SELECT version,np,MIN(total_time_s) AS best FROM perf_runs GROUP BY 1,2"
    ).fetchdf()
    con.close()

    t1 = float(df_best.loc[(df_best.version == "V1 Serial") & (df_best.np == 1), "best"].iloc[0])

    def _label(v: str) -> str:
        return v.replace("V2 2.1-broadcast-all", "V2 Broadcast")\
                .replace("V2 2.2-scatter-halo",  "V2 Scatter")

    if "speedup" in plots:
        for v in sorted(df_best.version.unique()):
            sub = df_best[df_best.version == v].sort_values("np")
            plt.figure()
            plt.plot(sub.np, t1 / sub.best, marker="o")
            plt.title(f"Speed-up – {_label(v)}")
            plt.xlabel("np"); plt.ylabel("Speed-up S")
            plt.grid(True, ls="--", lw=0.5)
            plt.savefig(PLOTS_DIR / f"speedup_{v.replace('/','-').replace(' ','_')}.png",
                        bbox_inches="tight")
            plt.close()
        CON.print(f"[green]✓ plots → {PLOTS_DIR}/speedup_*.png[/]")

    if "efficiency" in plots:
        rows = []
        for _, r in df_best.iterrows():
            rows.append((_label(r.version), r.np, (t1 / r.best) / r.np))
        df_e = pd.DataFrame(rows, columns=["version","np","E"])
        plt.figure(figsize=(8,4))
        plt.bar([f"{v}\nnp={n}" for v,n,_ in rows], df_e.E)
        plt.title("Strong-scaling efficiency")
        plt.ylabel("Efficiency E")
        plt.ylim(0,1.1); plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        plt.savefig(PLOTS_DIR / "efficiency.png", bbox_inches="tight")
        plt.close()
        CON.print(f"[green]✓ plot → {PLOTS_DIR}/efficiency.png[/]")

# ───────────────────────────── export ────────────────────────────────
@APP.command()
def export(
    view: str = Argument(..., help="Table/view name"),
    outfile: Path = Argument(..., help="Output .csv / .parquet"),
):
    con = ddb.connect(str(WAREHOUSE))
    try:
        df = con.execute(f"SELECT * FROM {view}").fetchdf()
    except Exception as e:
        CON.print(f"[red]✗ cannot select from '{view}':[/] {e}")
        sys.exit(1)
    finally:
        con.close()

    outfile.parent.mkdir(exist_ok=True, parents=True)
    if outfile.suffix.lower() == ".csv":
        df.to_csv(outfile, index=False)
    elif outfile.suffix.lower() == ".parquet":
        df.to_parquet(outfile, index=False, engine="pyarrow")
    else:
        CON.print("[red]✗ outfile must end with .csv or .parquet[/]")
        sys.exit(1)

    CON.print(f"[green]✓ exported {len(df):,} rows → {outfile}[/]")

# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    APP()
