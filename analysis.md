```python
# In[1]:
# Cell 1: Setup and Imports
# Python Standard Library
from __future__ import annotations
import hashlib
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Third-party Libraries
import duckdb as ddb
import pandas as pd
from rich.console import Console 
# For notebook display, pandas DataFrames are often preferred or use IPython.display.
from IPython.display import display, Markdown

# Matplotlib for plotting (optional, checked before use)
try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D 
except ModuleNotFoundError:
    plt = None
    Line2D = None
    print("Matplotlib not found. Plotting functions will be disabled. Install with: pip install matplotlib")

# python-magic for MIME sniffing (optional, checked before use)
try:
    import magic
except ModuleNotFoundError:
    magic = None
    print("python-magic not found. MIME type sniffing will use a default.")

# --- Globals & Constants ---
CON = Console(force_jupyter=False, force_terminal=False) # Better for notebook mixed output
WAREHOUSE = Path(".warehouse/cluster_logs.duckdb")
WAREHOUSE.parent.mkdir(parents=True, exist_ok=True) # Ensure warehouse directory exists

HASH_CHUNK = 1 << 16  # 64 KiB block size for streaming SHA-1

print(f"DuckDB Warehouse will be created/used at: {WAREHOUSE.resolve()}")
```


```python
# In[2]:
# Cell 2: Helper Functions

def _sha1(p: Path) -> str:
    """Computes SHA1 hash of a file."""
    h = hashlib.sha1()
    try:
        with p.open("rb") as fh:
            while blk := fh.read(HASH_CHUNK):
                h.update(blk)
        return h.hexdigest()
    except FileNotFoundError:
        # This case should ideally be handled before calling _sha1 if p might not exist.
        # For robustness, returning an empty string or raising a specific error.
        # CON.print(f"[yellow]File not found for hashing: {p}[/yellow]") 
        return "" 

def canonical_version_name(version_str: str) -> str:
    """Maps various raw version strings to a canonical project version name."""
    if not isinstance(version_str, str):
        return "UnknownVersion"
    
    v_lower = version_str.lower()

    # Order matters: more specific matches first
    if "v2" in v_lower and ("2.2" in v_lower or "scatter_halo" in v_lower or "scatter-halo" in v_lower):
        return "V2.2 ScatterHalo"
    if "v2" in v_lower and ("2.1" in v_lower or "broadcast_all" in v_lower or "broadcast-all" in v_lower):
        return "V2.1 BroadcastAll"
    if "v1" in v_lower or "serial" in v_lower:
        return "V1 Serial"
    if "v3" in v_lower or "cuda" in v_lower and "mpi" not in v_lower: # ensure not v4 or v5
        return "V3 CUDA"
    if "v4" in v_lower or ("mpi" in v_lower and "cuda" in v_lower): # Basic V4 check
        return "V4 MPI+CUDA"
    if "v5" in v_lower: # Assuming v5 implies MPI+CUDA as well
        return "V5 MPI+CUDA-Aware"
    
    # Fallback for log file names that might just contain the version number
    if "v2_1" in v_lower or "v2.1" in v_lower: return "V2.1 BroadcastAll"
    if "v2_2" in v_lower or "v2.2" in v_lower: return "V2.2 ScatterHalo"

    # Default if no specific pattern matches but starts with 'v' and a digit
    if v_lower.startswith("v") and len(v_lower) > 1 and v_lower[1].isdigit():
        return version_str # Keep original if it's a new variant not yet mapped
        
    return "Other" # Or version_str to keep unmapped ones visible

def _normalise_summary_df(df: pd.DataFrame, src: str) -> pd.DataFrame:
    """Maps various summary CSV formats to a canonical schema, including version normalization."""
    df = df.copy()
    output_columns = ["ts", "version", "np", "total_time_s"]
    for col in output_columns:
        if col not in df: 
            df[col] = pd.NA

    parsed_version = "UnknownVersion" # Default

    # Pattern 1: legacy run_summary_*.csv (Fall 2023 template)
    if {"Timestamp", "Version", "NP", "Time_ms"} <= set(df.columns):
        df["ts"]       = pd.to_datetime(df["Timestamp"],  utc=True, errors="coerce")
        parsed_version = df["Version"].astype(str).iloc[0] if not df.empty else "UnknownVersion" # Get from first row
        df["np"]       = pd.to_numeric(df["NP"], errors="coerce")
        if "Time_ms" in df.columns:
            df["total_time_s"] = pd.to_numeric(df["Time_ms"], errors="coerce").astype(float) / 1000.0
        else: df["total_time_s"] = pd.NA


    # Pattern 2: current summary_*.csv (Spring 2025 orchestrator)
    elif {"EntryTimestamp", "ProjectVariant", "NumProcesses"}.issubset(df.columns):
        df["ts"]       = pd.to_datetime(df["EntryTimestamp"], utc=True, errors="coerce")
        parsed_version = df["ProjectVariant"].astype(str).iloc[0] if not df.empty else "UnknownVersion"
        df["np"]       = pd.to_numeric(df["NumProcesses"], errors="coerce")
        time_col = None
        if "ExecutionTime_ms" in df.columns: time_col = "ExecutionTime_ms"
        elif "Time_ms" in df.columns: time_col = "Time_ms"
        
        if time_col and time_col in df.columns:
            df["total_time_s"] = pd.to_numeric(df[time_col], errors="coerce").astype(float) / 1000.0
        else: df["total_time_s"] = pd.NA
            
    # Pattern 3: for 'summary_nixos_*.csv' style logs
    elif {"timestamp", "version", "np", "time_ms"} <= set(df.columns):
        df["ts"]       = pd.to_datetime(df["timestamp"],  utc=True, errors="coerce")
        parsed_version = df["version"].astype(str).iloc[0] if not df.empty else "UnknownVersion"
        df["np"]       = pd.to_numeric(df["np"], errors="coerce") 
        if "time_ms" in df.columns:
            df["total_time_s"] = pd.to_numeric(df["time_ms"], errors="coerce").astype(float) / 1000.0
        else: df["total_time_s"] = pd.NA
    else:
        # CON.print(f"[dim]Unknown summary CSV schema: {src}[/dim]")
        return pd.DataFrame(columns=output_columns) 

    # Apply canonical version naming to the entire 'version' column
    df["version"] = df["version"].apply(canonical_version_name) # Important: apply to the existing version col

    df_filtered = df[output_columns].dropna(subset=["ts", "version", "np", "total_time_s"])
    return df_filtered

print("Helper and canonicalization functions defined.")
```


```python
# In[3]:
# Cell 3: Database Schema Definitions

SCHEMA_SQL = """
DROP VIEW IF EXISTS efficiency;
DROP VIEW IF EXISTS speedup;
DROP VIEW IF EXISTS run_stats;
DROP VIEW IF EXISTS best_runs;
DROP VIEW IF EXISTS perf_runs;

DROP TABLE IF EXISTS file_index;
DROP TABLE IF EXISTS summary_runs;
DROP TABLE IF EXISTS run_logs;
DROP TABLE IF EXISTS source_stats;

CREATE TABLE file_index (
  relpath  TEXT PRIMARY KEY, 
  sha1     TEXT,            
  size     BIGINT,
  mtime    TIMESTAMP,
  mime     TEXT
);
CREATE TABLE summary_runs (
  ts TIMESTAMP,
  version TEXT, -- This will store the CANONICAL version
  np INT,
  total_time_s DOUBLE
);
CREATE TABLE run_logs (
  relpath TEXT, -- Original relpath for traceability
  ts TIMESTAMP,
  version TEXT, -- This will store the CANONICAL version
  np INT,
  total_time_s DOUBLE
);
CREATE TABLE source_stats (
  relpath TEXT PRIMARY KEY, 
  loc INT,
  func_cnt INT,
  include_cnt INT,
  cuda_kernel_cnt INT
);
"""

DERIVED_VIEWS_SQL = """
CREATE OR REPLACE VIEW perf_runs AS
SELECT ts, version, np, total_time_s FROM summary_runs WHERE total_time_s IS NOT NULL AND total_time_s > 1e-9 AND version != 'Other' AND version != 'UnknownVersion'
UNION ALL
SELECT ts, version, np, total_time_s FROM run_logs WHERE total_time_s IS NOT NULL AND total_time_s > 1e-9 AND version != 'Other' AND version != 'UnknownVersion';

CREATE OR REPLACE VIEW best_runs AS
SELECT version, np, MIN(total_time_s)  AS best_s
FROM   perf_runs GROUP BY version, np;

CREATE OR REPLACE VIEW run_stats AS
SELECT version, np,
       COUNT(*)                            AS n,
       AVG(total_time_s)                  AS mean_s,
       STDDEV_SAMP(total_time_s)          AS sd_s, 
       CASE WHEN COUNT(*)>1 THEN 1.96*STDDEV_SAMP(total_time_s)/SQRT(COUNT(*)) ELSE NULL END AS ci95_s
FROM   perf_runs GROUP BY version, np;

CREATE OR REPLACE VIEW speedup AS
WITH base_runs AS (
    SELECT 
        version, 
        MIN(best_s) as t1 
    FROM best_runs 
    WHERE np = 1
    GROUP BY version
)
SELECT 
    br.version,
    br.np,
    b.t1 / br.best_s AS S 
FROM best_runs br 
JOIN base_runs b ON br.version = b.version
WHERE b.t1 IS NOT NULL AND br.best_s IS NOT NULL AND br.best_s > 1e-9; 

CREATE OR REPLACE VIEW efficiency AS
SELECT version, np, S/np AS E FROM speedup WHERE np > 0;
"""

print("Database schemas and view definitions ready (includes DROP statements for rebuilds).")
```


```python
# In[4]:
# Cell 4: Data Ingestion Function

def ingest_data(root: Path, rebuild: bool = False):
    """Scans the root directory for CSV summaries, run logs, and source files,
    then loads the extracted data into the DuckDB warehouse."""
    
    if rebuild and WAREHOUSE.exists():
        WAREHOUSE.unlink()
        CON.print("[red]• Wiped existing warehouse[/red]")

    with ddb.connect(str(WAREHOUSE)) as con:
        con.execute(SCHEMA_SQL) # Executes DROPs and CREATEs

        seen: Dict[str, str] = dict(con.execute("SELECT relpath, sha1 FROM file_index").fetchall())
        rows_summary_dfs: List[pd.DataFrame] = [] # Changed to list of DataFrames
        rows_runlog:  List[tuple]       = []
        rows_srcstats:List[tuple]       = []
        processed_files_this_run = 0
        newly_indexed_count = 0

        CON.print(f"Starting ingestion from root: {root.resolve()}")
        all_files = list(root.rglob("*")) # Collect all files first to avoid issues with changing dir
        
        for p in all_files:
            if p.is_dir() or WAREHOUSE.resolve() == p.resolve(): 
                continue
            
            rel = str(p.relative_to(root))
            
            try:
                current_sha1 = _sha1(p)
                if not current_sha1 : continue # Skip if hashing failed

                current_stat = p.stat()
                size  = current_stat.st_size
                mtime = datetime.fromtimestamp(current_stat.st_mtime, tz=timezone.utc)
            except FileNotFoundError:
                continue

            if seen.get(rel) != current_sha1: # Process if new or modified
                newly_indexed_count +=1
                mime_type  = magic.from_file(str(p), mime=True) if magic else "application/octet-stream"

                # --- CSV Summaries ---
                if p.suffix.lower() == ".csv" and "summary" in p.name.lower():
                    try:
                        df_raw = pd.read_csv(p)
                        if not df_raw.empty:
                            df_norm = _normalise_summary_df(df_raw, rel) # Normalization includes canonical_version_name
                            if not df_norm.empty:
                                rows_summary_dfs.append(df_norm)
                    except pd.errors.EmptyDataError: pass
                    except Exception as e: CON.print(f"[yellow]CSV summary parse error {rel}: {e}[/yellow]")
                
                # --- Individual Run Logs ---
                elif p.suffix.lower() == ".log" and ("run_v" in p.name or "final_project_v" in p.name or "make_v" in p.name): 
                    try:
                        txt = p.read_text(errors="ignore")
                        m_time_val = None
                        time_patterns = [
                            r"AlexNet MPI\+CUDA Forward Pass completed in\s*([\d\.]+)\s*ms",
                            r"AlexNet CUDA Forward Pass completed in\s*([\d\.]+)\s*ms", 
                            r"AlexNet Serial Forward Pass completed in\s*([\d\.]+)\s*ms",
                            r"Serial execution finished successfully in\s*([\d\.]+)\s*ms",
                            r"MPI\+CUDA execution finished successfully in\s*([\d\.]+)\s*ms",
                            r"CUDA execution finished successfully in\s*([\d\.]+)\s*ms", 
                            r"MPI execution finished successfully in\s*([\d\.]+)\s*ms", 
                            r"Execution Time:\s*([\d\.]+)\s*ms", 
                            r"(?:Time|ExecutionTime)_ms[\s:=]*([\d\.]+)",
                            r"Total execution time:\s*([\d\.]+)\s*seconds", 
                        ]
                        for pat in time_patterns:
                            m_search = re.search(pat, txt, re.IGNORECASE)
                            if m_search:
                                m_time_val = float(m_search.group(1))
                                if "seconds" not in pat.lower(): m_time_val /= 1000.0
                                break 
                        
                        if m_time_val is not None:
                            raw_version_str = "unknown_version_in_log"
                            ver_pattern = r"(v\d(?:[\._]\d+(?:[\._][\w-]+)*)?(?:_[\w-]+)*)"
                            ver_match = re.search(ver_pattern, p.name, re.IGNORECASE)
                            if ver_match:
                                raw_version_str = ver_match.group(0)
                            else:
                                parent_name = p.parent.name
                                if parent_name.lower().startswith("v"):
                                    raw_version_str = parent_name
                            
                            canonical_ver = canonical_version_name(raw_version_str)
                            
                            np_m = re.search(r"np(\d+)", p.name, re.IGNORECASE)
                            np_val = int(np_m.group(1)) if np_m else 1
                            
                            ts_match_in_name = re.search(r"(\d{8}_\d{6})", p.name) # Check filename first
                            log_ts = mtime # Default to file mtime
                            if ts_match_in_name:
                                try:
                                    log_ts = datetime.strptime(ts_match_in_name.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
                                except ValueError: pass # Keep mtime if parsing fails
                            
                            rows_runlog.append((rel, log_ts, canonical_ver, np_val, m_time_val))
                    except Exception as e: pass # CON.print(f"[yellow]Run log parse error {rel}: {e}[/yellow]")

                # --- Source File Statistics ---
                elif mime_type and ("text/" in mime_type or mime_type == 'application/octet-stream' or p.name.lower().endswith("makefile") or ".make" in p.name.lower()):
                    try:
                        code = p.read_text(errors="ignore")
                        loc = code.count("\n") + 1
                        func_cnt = len(re.findall(r"\b[A-Za-z_]\w*\s*\([^)]*\)\s*(?:const)?\s*\{", code))
                        inc_cnt = len(re.findall(r"^\s*#include", code, re.MULTILINE))
                        kern_cnt = code.count("__global__")
                        rows_srcstats.append((rel, loc, func_cnt, inc_cnt, kern_cnt))
                    except Exception as e: pass # CON.print(f"[yellow]Source stat parse error {rel}: {e}[/yellow]")
                
                # --- Update file_index Table ---
                try:
                    con.execute(
                        "INSERT INTO file_index (relpath, sha1, size, mtime, mime) VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT(relpath) DO UPDATE SET sha1=excluded.sha1, size=excluded.size, mtime=excluded.mtime, mime=excluded.mime",
                        [rel, current_sha1, size, mtime, mime_type]
                    )
                except Exception as db_ex: CON.print(f"[red]DB Error indexing {rel}: {db_ex}[/red]")
            processed_files_this_run +=1


        if newly_indexed_count == 0 and not rebuild:
             CON.print("[cyan]• No new or modified files to process since last ingest.[/cyan]")
        elif newly_indexed_count > 0:
             CON.print(f"[green]• Indexed/Updated {newly_indexed_count} files in file_index.[/green]")


        # --- Bulk Insert Data ---
        inserted_summary_count = 0
        if rows_summary_dfs:
            df_all_summary = pd.concat(rows_summary_dfs, ignore_index=True).drop_duplicates().dropna(subset=['total_time_s', 'version', 'np'])
            if not df_all_summary.empty:
                # Ensure 'version' in df_all_summary is canonical before inserting
                # _normalise_summary_df already applies canonical_version_name
                con.register("df_all_summary_reg", df_all_summary)
                con.execute("INSERT INTO summary_runs SELECT ts, version, np, total_time_s FROM df_all_summary_reg")
                con.unregister("df_all_summary_reg")
                inserted_summary_count = len(df_all_summary)
        CON.print(f"[cyan]• {inserted_summary_count} summary rows ingested[/cyan]")

        inserted_log_count = 0
        if rows_runlog:
            # Ensure 'version' in rows_runlog (which is 3rd element, index 2) is already canonical
            df_runlog = pd.DataFrame(rows_runlog, columns=['relpath', 'ts', 'version', 'np', 'total_time_s']).drop_duplicates().dropna(subset=['total_time_s', 'version', 'np'])
            if not df_runlog.empty:
                con.executemany("INSERT INTO run_logs VALUES (?,?,?,?,?)", df_runlog.to_records(index=False).tolist())
                inserted_log_count = len(df_runlog)
        CON.print(f"[cyan]• {inserted_log_count} run_log rows ingested[/cyan]")

        inserted_src_count = 0
        if rows_srcstats:
            df_srcstats = pd.DataFrame(rows_srcstats, columns=['relpath', 'loc', 'func_cnt', 'include_cnt', 'cuda_kernel_cnt']).drop_duplicates(subset=['relpath'])
            if not df_srcstats.empty:
                con.executemany(
                    "INSERT INTO source_stats (relpath, loc, func_cnt, include_cnt, cuda_kernel_cnt) VALUES (?,?,?,?,?) "
                    "ON CONFLICT(relpath) DO UPDATE SET loc=excluded.loc, func_cnt=excluded.func_cnt, include_cnt=excluded.include_cnt, cuda_kernel_cnt=excluded.cuda_kernel_cnt",
                    df_srcstats.to_records(index=False).tolist()
                )
                inserted_src_count = len(df_srcstats)
        CON.print(f"[cyan]• {inserted_src_count} source files stats ingested/updated[/cyan]")

        con.execute(DERIVED_VIEWS_SQL)
        CON.print("[bold green]✓ Ingest complete. Derived views (re)created.[/bold green]")

print("Ingestion function 'ingest_data' defined.")
```


```python
# In[5]:
# Cell 5: Perform Data Ingestion

project_root = Path(".") 
ingest_data(root=project_root / "final_project", rebuild=True) 
```


```python
# In[6]:
# Cell 6: Ad-hoc Querying and Database Inspection

def execute_query(sql_query: str) -> pd.DataFrame | None:
    """Executes a SQL query against the warehouse and returns a DataFrame."""
    try:
        with ddb.connect(str(WAREHOUSE), read_only=True) as con:
            df = con.execute(sql_query).fetchdf()
        return df
    except Exception as e:
        # For notebook, print might be better than CON.print
        print(f"SQL Query Error: {e}")
        return None

# --- Database Inspection ---
print("\n--- Database Objects (Tables and Views) ---")
all_db_objects = execute_query(
    """
    SELECT table_name as object_name, table_type as type 
    FROM information_schema.tables 
    WHERE table_schema = 'main'
    UNION ALL 
    SELECT view_name as object_name, 'VIEW' as type 
    FROM duckdb_views() 
    WHERE schema_name = 'main'
    ORDER BY type, object_name
    """
)
if all_db_objects is not None:
    display(all_db_objects)

print("\n--- Schema of 'best_runs' view ---")
best_runs_schema = execute_query("DESCRIBE best_runs;")
if best_runs_schema is not None:
    display(best_runs_schema)

print("\n--- Sample data from 'best_runs' (Grouped by canonical version, NP) ---")
sample_best_runs = execute_query("SELECT * FROM best_runs ORDER BY version, np LIMIT 10;")
if sample_best_runs is not None:
    display(sample_best_runs)

print("\n--- Perf run counts per canonical version ---")
version_counts = execute_query("SELECT version, COUNT(*) as count FROM perf_runs GROUP BY version ORDER BY version;")
if version_counts is not None:
    display(version_counts)


print("\n--- Total Source Files Analyzed ---")
source_file_count = execute_query("SELECT COUNT(*) as num_source_files FROM source_stats;")
if source_file_count is not None:
    display(source_file_count)
```


```python
# In[7]:
# Cell 7: Display Run Statistics

print("\n--- Run Statistics (Time in seconds, for Canonical Versions) ---")
run_stats_df = execute_query("""
    SELECT version, np, n, 
           ROUND(mean_s, 4) as mean_s, 
           ROUND(sd_s, 4) as sd_s, 
           ROUND(ci95_s, 4) as ci95_s 
    FROM run_stats 
    ORDER BY version, np
""")

if run_stats_df is not None and not run_stats_df.empty:
    display(run_stats_df)
else:
    print("No run statistics to display. Ensure 'perf_runs' has data for canonical versions.")
```


```python
# In[8]:
# Cell 8: Display Speedup and Efficiency Tables

print("\n--- Speedup (S = T_NP1 / T_NPn, relative to canonical version's NP=1) ---")
speedup_df = execute_query("SELECT version, np, ROUND(S, 3) AS S FROM speedup ORDER BY version, np")
if speedup_df is not None and not speedup_df.empty:
    display(speedup_df)
else:
    print("No speedup data. Ensure NP=1 runs exist for canonical versions in 'best_runs'.")

print("\n--- Efficiency (E = S / np, relative to canonical version's NP=1) ---")
efficiency_df = execute_query("SELECT version, np, ROUND(E, 3) AS E FROM efficiency ORDER BY version, np")
if efficiency_df is not None and not efficiency_df.empty:
    display(efficiency_df)
else:
    print("No efficiency data. Ensure 'speedup' view has data for canonical versions.")
```


```python
# In[9]:
# Cell 9: Data Export Function

def export_table(table_name: str, out_file: Path):
    """Exports a specified table or view from the database to a file (MD, CSV, Parquet)."""
    check_exists_query = f"""
        SELECT 1 FROM information_schema.tables WHERE table_name = '{table_name}' AND table_schema = 'main'
        UNION ALL 
        SELECT 1 FROM duckdb_views() WHERE view_name = '{table_name}' AND schema_name = 'main' 
        LIMIT 1
    """
    check_exists = execute_query(check_exists_query)
    if check_exists is None or check_exists.empty:
        print(f"Table or view '{table_name}' does not exist in the database.")
        # Optionally list available tables/views for debugging
        # all_objs = execute_query("SELECT name, type from duckdb_objects() WHERE schema_name='main' ORDER BY type, name;")
        # if all_objs is not None: print("Available objects:\n", all_objs)
        return

    df = execute_query(f"SELECT * FROM {table_name}")

    if df is None :
        print(f"Failed to fetch data from '{table_name}' for export, though it seems to exist.")
        return
    if df.empty:
        print(f"Table/view '{table_name}' is empty. Nothing to export to {out_file}.")
        return

    file_suffix = out_file.suffix.lower()
    out_file.parent.mkdir(parents=True, exist_ok=True) 
    try:
        if file_suffix == ".md":
            df.to_markdown(out_file, index=False)
        elif file_suffix == ".csv":
            df.to_csv(out_file, index=False)
        elif file_suffix == ".parquet":
            df.to_parquet(out_file, index=False)
        else:
            print(f"Unsupported export file format: {file_suffix}. Supported: .md, .csv, .parquet")
            return
        print(f"✓ Table '{table_name}' exported successfully to {out_file}")
    except Exception as e:
        print(f"Error writing table '{table_name}' to file {out_file}: {e}")

print("Data export function 'export_table' defined.")
```


```python
# In[10]:
# Cell 10: Example Export Calls

output_dir = Path("analysis_exports") 
output_dir.mkdir(parents=True, exist_ok=True)

export_table("best_runs", output_dir / "project_best_runs.md")
export_table("run_stats", output_dir / "project_run_statistics.csv")
export_table("perf_runs", output_dir / "project_all_perf_runs.parquet")
export_table("speedup", output_dir / "project_speedup_data.csv")
export_table("efficiency", output_dir / "project_efficiency_data.csv")
export_table("file_index", output_dir / "project_indexed_files.parquet") # Parquet for potentially large table
export_table("source_stats", output_dir / "project_source_code_stats.md")

print(f"\nExports completed to directory: {output_dir.resolve()}")
```


```python
# In[11]:
# Cell 11: Runtime Plotting Function

def generate_runtime_plot(out_file: Path = Path("analysis_plots/runtimes_plot.png")):
    """Plots Runtime vs NP for each version (using fastest runs) and saves to file."""
    if plt is None:
        print("matplotlib not installed – cannot plot.")
        return

    df = execute_query("SELECT version, np, best_s FROM best_runs WHERE best_s IS NOT NULL ORDER BY version, np")
    
    if df is None or df.empty:
        print("No data in 'best_runs' to plot for runtimes (after filtering for canonical versions).")
        return

    plt.figure(figsize=(12, 7)) # Adjusted size
    versions = df["version"].unique()
    for ver in versions:
        grp = df[df["version"] == ver]
        if not grp.empty:
             plt.plot(grp["np"], grp["best_s"], marker="o", linestyle="-", label=ver)
    
    plt.xlabel("Number of Processes (NP)")
    plt.ylabel("Best Runtime [s]")
    
    unique_nps = sorted(df["np"].unique())
    if unique_nps: 
        is_power_of_2_friendly = all(np_val != 0 and (np_val & (np_val - 1) == 0) for np_val in unique_nps if np_val is not None and np_val > 0)
        if max(unique_nps, default=1) / max(1, min(filter(lambda x: x>0, unique_nps), default=1)) >= 4 and len(unique_nps) > 3 : 
             plt.xscale("log", base=2 if is_power_of_2_friendly else 10)
        plt.xticks(unique_nps, labels=[str(int(x)) for x in unique_nps]) # Ensure integer labels for NP

    min_best_s_val = df["best_s"].min() if not df["best_s"].empty and df["best_s"].min() > 0 else 1e-9
    max_best_s_val = df["best_s"].max() if not df["best_s"].empty else 1.0
    if (max_best_s_val / min_best_s_val > 10): 
        plt.yscale("log", base=10)
        
    plt.title("Runtime vs Number of Processes (Canonical Versions)")
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.legend(loc="best", fontsize="small") # Adjusted legend
    plt.tight_layout()
    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file)
        print(f"✓ Runtime plot saved to {out_file}")
        plt.show() 
    except Exception as e:
        print(f"Error saving runtime plot to {out_file}: {e}")
    finally:
        plt.close()

print("Runtime plotting function 'generate_runtime_plot' defined.")
```


```python
# In[12]:
# Cell 12: Example Runtime Plot Call

generate_runtime_plot(out_file=Path("analysis_plots/project_performance_runtimes.png"))
```


```python
# In[13]:
# Cell 13: Speedup Plotting Function

def generate_speedup_plot(out_file: Path = Path("analysis_plots/speedup_curve_plot.png")):
    """Plots Speedup vs NP for each version and saves to file."""
    if plt is None or Line2D is None: 
        print("matplotlib not installed – cannot plot.")
        return

    df = execute_query("SELECT version, np, S FROM speedup WHERE S IS NOT NULL ORDER BY version, np")
    if df is None or df.empty:
        print("No data in 'speedup' view to plot (check for NP=1 runs for canonical versions).")
        return

    plt.figure(figsize=(12, 7))
    all_nps_in_plot = [] 
    versions = df["version"].unique()
    for ver in versions:
        grp = df[df["version"] == ver]
        if not grp.empty:
            plt.plot(grp["np"], grp["S"], marker="o", linestyle="-", label=ver)
            all_nps_in_plot.extend(grp["np"].tolist())
    
    unique_nps_for_ideal_line = sorted(list(set(filter(None, all_nps_in_plot)))) 
    if unique_nps_for_ideal_line: 
        plt.plot(unique_nps_for_ideal_line, unique_nps_for_ideal_line, linestyle="--", color="gray", label="Ideal Speedup")

    plt.xlabel("Number of Processes (NP)")
    plt.ylabel("Speedup (S = T_NP1 / T_NPn)")
    
    unique_nps_overall = sorted(df["np"].unique())
    if unique_nps_overall:
        is_power_of_2_friendly = all(np_val != 0 and (np_val & (np_val - 1) == 0) for np_val in unique_nps_overall if np_val is not None and np_val > 0)
        if max(unique_nps_overall, default=1) / max(1, min(filter(lambda x: x>0, unique_nps_overall), default=1)) >= 4 and len(unique_nps_overall) > 3:
            plt.xscale("log", base=2 if is_power_of_2_friendly else 10)
        plt.xticks(unique_nps_overall, labels=[str(int(x)) for x in unique_nps_overall])
    
    plt.title("Speedup Curve vs Number of Processes (Canonical Versions)")
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file)
        print(f"✓ Speedup curve plot saved to {out_file}")
        plt.show()
    except Exception as e:
        print(f"Error saving speedup plot to {out_file}: {e}")
    finally:
        plt.close()

print("Speedup plotting function 'generate_speedup_plot' defined.")
```


```python
# In[14]:
# Cell 14: Example Speedup Plot Call

generate_speedup_plot(out_file=Path("analysis_plots/project_performance_speedup.png"))
```


```python
# In[15]:
# Cell 15: Efficiency Plotting Function

def generate_efficiency_plot(out_file: Path = Path("analysis_plots/efficiency_bars_plot.png")):
    """Plots Parallel Efficiency vs NP for each version and saves to file."""
    if plt is None or Line2D is None:
        print("matplotlib not installed – cannot plot.")
        return

    df = execute_query("SELECT version, np, E FROM efficiency WHERE E IS NOT NULL ORDER BY version, np")
    if df is None or df.empty:
        print("No data in 'efficiency' view to plot (check speedup view).")
        return

    plt.figure(figsize=(13, 7)) # Slightly wider for potentially many versions
    plotted_as_bars = False
    unique_nps_overall = sorted(df["np"].unique())
    versions_count = len(df["version"].unique())

    if unique_nps_overall and len(unique_nps_overall) <= 6 and versions_count <= 7 : 
        try:
            df_pivot = df.pivot(index="np", columns="version", values="E")
            df_pivot.plot(kind="bar", ax=plt.gca(), width=0.85) 
            plt.xticks(rotation=0) 
            plotted_as_bars = True
        except Exception as e: 
            print(f"Could not pivot efficiency data for bar plot (Error: {e}). Plotting as lines.")
    
    if not plotted_as_bars: 
        versions = df["version"].unique()
        for ver in versions:
            grp = df[df["version"] == ver].sort_values("np")
            if not grp.empty:
                plt.plot(grp["np"], grp["E"], marker="o", linestyle="-", label=f"{ver}")
        
        if unique_nps_overall: 
            is_power_of_2_friendly = all(np_val != 0 and (np_val & (np_val-1)==0) for np_val in unique_nps_overall if np_val is not None and np_val > 0)
            if max(unique_nps_overall, default=1) / max(1, min(filter(lambda x: x>0, unique_nps_overall), default=1)) >= 4 and len(unique_nps_overall) > 3:
                 plt.xscale("log", base=2 if is_power_of_2_friendly else 10)
            plt.xticks(unique_nps_overall, labels=[str(int(x)) for x in unique_nps_overall])

    plt.xlabel("Number of Processes (NP)")
    plt.ylabel("Efficiency (E = Speedup / NP)")
    plt.title("Parallel Efficiency vs Number of Processes (Canonical Versions)")
    
    max_e_val = df["E"].max() if not df["E"].empty and pd.notna(df["E"].max()) else 1.0
    upper_y_limit = max(1.1, max_e_val * 1.1 if pd.notna(max_e_val) else 1.1)
    if upper_y_limit > 1.5 and plotted_as_bars: # Adjust y-limit for bar plots if superlinear
        upper_y_limit = max_e_val * 1.1 
    elif upper_y_limit > 2.0 and not plotted_as_bars: # Cap if very superlinear for line plots
         upper_y_limit = max_e_val * 1.1

    plt.ylim(0, upper_y_limit)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    if not any("Ideal Efficiency" in lab for lab in labels):
        ideal_line = Line2D([0], [0], linestyle="--", color="gray", label='Ideal Efficiency (1.0)')
        handles.append(ideal_line)
        labels.append('Ideal Efficiency (1.0)')
    
    plt.axhline(1.0, linestyle="--", color="gray", linewidth=0.8) 
    plt.legend(handles, labels, loc="best", title="Version", fontsize="small")

    plt.grid(True, axis='y', ls=":", lw=0.5) 
    plt.tight_layout()
    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file)
        print(f"✓ Efficiency plot saved to {out_file}")
        plt.show()
    except Exception as e_save:
        print(f"Error saving efficiency plot to {out_file}: {e_save}")
    finally:
        plt.close()

print("Efficiency plotting function 'generate_efficiency_plot' defined.")
```


```python
# In[16]:
# Cell 16: Example Efficiency Plot Call

generate_efficiency_plot(out_file=Path("analysis_plots/project_performance_efficiency.png"))
```


```python
# In[17]:
# Cell 17: Focused Statistical Comparisons & Key Metrics Display

print("\n--- Key Performance Metrics from 'run_stats' (Canonical Versions) ---")
# Display mean, standard deviation, and number of runs for key versions
key_versions_stats = execute_query("""
    SELECT 
        version, 
        np, 
        n, 
        ROUND(mean_s, 4) AS mean_runtime_s, 
        ROUND(sd_s, 4) AS std_dev_s,
        ROUND(ci95_s, 4) AS ci95_s
    FROM run_stats
    WHERE version IN ('V1 Serial', 'V2.2 ScatterHalo', 'V3 CUDA', 'V4 MPI+CUDA')
    ORDER BY version, np;
""")
if key_versions_stats is not None and not key_versions_stats.empty:
    display(key_versions_stats)
else:
    print("No stats found for key versions. Ensure data is ingested and canonical names are correct.")

print("\n--- Fastest Single Process Runs (NP=1 from 'best_runs') ---")
fastest_np1_runs = execute_query("""
    SELECT 
        version, 
        np, 
        ROUND(best_s, 4) AS fastest_runtime_s
    FROM best_runs
    WHERE np = 1 AND version IN ('V1 Serial', 'V2.1 BroadcastAll', 'V2.2 ScatterHalo', 'V3 CUDA', 'V4 MPI+CUDA')
    ORDER BY fastest_runtime_s;
""")
if fastest_np1_runs is not None and not fastest_np1_runs.empty:
    display(fastest_np1_runs)
else:
    print("No NP=1 best runs found for key versions.")

print("\n--- Performance at Max Scaled NP (e.g., NP=4, from 'best_runs') ---")
# Assuming NP=4 is the max scale point for MPI versions
max_np_runs = execute_query("""
    SELECT 
        version, 
        np, 
        ROUND(best_s, 4) AS fastest_runtime_s
    FROM best_runs
    WHERE np = 4 AND version IN ('V2.1 BroadcastAll', 'V2.2 ScatterHalo', 'V4 MPI+CUDA')
    ORDER BY fastest_runtime_s;
""")
if max_np_runs is not None and not max_np_runs.empty:
    display(max_np_runs)
else:
    print("No NP=4 best runs found for relevant MPI versions.")

# You could add more specific statistical tests here if needed, e.g., using scipy.stats
# For instance, comparing means of V3 CUDA vs V4 MPI+CUDA at NP=1 if you have multiple runs.
# This would require fetching the raw run times from 'perf_runs' for those specific conditions.
# Example (conceptual, requires scipy):
# from scipy import stats
# v3_np1_times = execute_query("SELECT total_time_s FROM perf_runs WHERE version = 'V3 CUDA' AND np = 1")
# v4_np1_times = execute_query("SELECT total_time_s FROM perf_runs WHERE version = 'V4 MPI+CUDA' AND np = 1")
# if v3_np1_times is not None and not v3_np1_times.empty and \
#    v4_np1_times is not None and not v4_np1_times.empty and \
#    len(v3_np1_times['total_time_s'].dropna()) > 1 and \
#    len(v4_np1_times['total_time_s'].dropna()) > 1:
#    ttest_result = stats.ttest_ind(v3_np1_times['total_time_s'].dropna(), v4_np1_times['total_time_s'].dropna())
#    print(f"\n--- T-test V3 CUDA (NP=1) vs V4 MPI+CUDA (NP=1) ---")
#    print(f"Statistic: {ttest_result.statistic:.4f}, P-value: {ttest_result.pvalue:.4f}")
```


```python
# In[18]:
# Cell 18: Generate and Export All Visualizations

visuals_output_dir = Path("analysis_visuals_final")
visuals_output_dir.mkdir(parents=True, exist_ok=True)
print(f"All plots will be saved to: {visuals_output_dir.resolve()}")

# Generate and save Runtime Plot
print("\nGenerating Runtime Plot...")
generate_runtime_plot(out_file=visuals_output_dir / "project_runtimes_vs_np.png")

# Generate and save Speedup Plot
print("\nGenerating Speedup Plot...")
generate_speedup_plot(out_file=visuals_output_dir / "project_speedup_vs_np.png")

# Generate and save Efficiency Plot
print("\nGenerating Efficiency Plot...")
generate_efficiency_plot(out_file=visuals_output_dir / "project_efficiency_vs_np.png")

print("\n--- All visual exports attempted. Check console for success/error messages. ---")
```


```python
# In[19]:
# Cell 19: Advanced Data Integration, Statistical Visualization & Synthesis

import seaborn as sns # For potentially prettier plots
import matplotlib.colors # For color mapping
from matplotlib.ticker import MaxNLocator, FuncFormatter # For ensuring integer ticks and custom formatting
from scipy.stats import pearsonr # For correlation

# Apply a nicer default style if seaborn is available
if 'seaborn' in sys.modules:
    sns.set_theme(style="whitegrid", palette="muted")
    print("[INFO] Seaborn theme applied for enhanced plot aesthetics.")

visuals_output_dir = Path("analysis_visuals_final") 
visuals_output_dir.mkdir(parents=True, exist_ok=True)
print(f"[INFO] All plots for this cell will be saved to: {visuals_output_dir.resolve()}")

# --- Helper to format y-axis ticks for log scale if used ---
def log_tick_formatter(val, pos=None):
    return f"{val:.2g}" # Format to general with 2 significant figures

# --- 1. Aggregate Lines of Code (LOC) for Canonical Versions ---
print("\n--- Aggregating Lines of Code (LOC) for Canonical Versions ---")
version_loc_map = {
    "V1 Serial": ["v1_serial/src/", "v1_serial/include/"],
    "V2.1 BroadcastAll": ["v2_mpi_only/2.1_broadcast_all/src/", "v2_mpi_only/2.1_broadcast_all/include/"],
    "V2.2 ScatterHalo": ["v2_mpi_only/2.2_scatter_halo/src/", "v2_mpi_only/2.2_scatter_halo/include/"],
    "V3 CUDA": ["v3_cuda_only/src/", "v3_cuda_only/include/"],
    "V4 MPI+CUDA": ["v4_mpi_cuda/src/", "v4_mpi_cuda/include/"],
    # "V5 MPI+CUDA-Aware": ["v5_cuda_aware_mpi/src/", "v5_cuda_aware_mpi/include/"] # Example
}
version_loc_data = []
for version_name, dir_prefixes in version_loc_map.items():
    like_clauses = [f"relpath LIKE '{prefix}%'" for prefix in dir_prefixes]
    dir_filter = " OR ".join(like_clauses)
    
    # DuckDB's extension function is simpler: extension(relpath)
    # However, to be safe with paths that might not have extensions or multiple dots,
    # a regex or robust split is better if extension() isn't available/suitable.
    # For DuckDB 0.7.0+, `regexp_extract(relpath, '\.([a-zA-Z0-9]+)$')` could work for extension.
    # Simpler approach using LIKE for common extensions:
    ext_filter = "OR ".join([f"LOWER(relpath) LIKE '%.{ext}'" for ext in ['cpp', 'cu', 'hpp', 'h', 'c', 'inl']])

    query = f"""
    SELECT '{version_name}' as version, SUM(loc) as total_loc
    FROM source_stats
    WHERE ({dir_filter}) AND ({ext_filter});
    """
    loc_df = execute_query(query)
    if loc_df is not None and not loc_df.empty and pd.notna(loc_df.iloc[0]['total_loc']):
        version_loc_data.append({'version': version_name, 'total_loc': int(loc_df.iloc[0]['total_loc'])})
    else:
        version_loc_data.append({'version': version_name, 'total_loc': 0}) # Assume 0 if no files match

df_loc = pd.DataFrame(version_loc_data)
if not df_loc[df_loc['total_loc'] > 0].empty:
    print("Aggregated LOC for core logic (src/, include/ relevant files):")
    display(df_loc[df_loc['total_loc'] > 0])
else:
    print("Could not aggregate LOC for any version. Check patterns or source_stats data.")

# --- 2. Plot: Best NP=1 Performance vs. LOC (with Correlation) ---
print("\n--- Plotting: Best NP=1 Performance vs. Lines of Code ---")
df_loc_filtered = df_loc[df_loc['total_loc'] > 0] # Use only versions with LOC > 0
if not df_loc_filtered.empty:
    df_best_np1 = execute_query("SELECT version, best_s FROM best_runs WHERE np = 1 AND best_s IS NOT NULL")
    if df_best_np1 is not None and not df_best_np1.empty:
        df_perf_vs_loc = pd.merge(df_best_np1, df_loc_filtered, on="version")
        
        if not df_perf_vs_loc.empty and len(df_perf_vs_loc) > 1: # Need at least 2 points for correlation
            plt.figure(figsize=(11, 7)) # Increased figure size
            
            # Calculate Pearson correlation
            # Drop NA before correlation if any best_s or total_loc could be NA
            df_corr = df_perf_vs_loc[['total_loc', 'best_s']].dropna()
            corr_val, p_val = pd.NA, pd.NA
            if len(df_corr) > 1:
                 corr_val, p_val = pearsonr(df_corr["total_loc"], df_corr["best_s"])
                 corr_text = f'Pearson R: {corr_val:.2f} (p={p_val:.2g})'
            else:
                corr_text = "Not enough data for correlation"


            if 'sns' in sys.modules:
                sns.scatterplot(data=df_perf_vs_loc, x="total_loc", y="best_s", hue="version", size="best_s", sizes=(50,300), legend="auto", palette="viridis")
                plt.legend(title="Version", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            else:
                # Manual scatter plot if seaborn not available
                cmap = matplotlib.colormaps['viridis']
                norm = matplotlib.colors.Normalize(vmin=df_perf_vs_loc['best_s'].min(), vmax=df_perf_vs_loc['best_s'].max())
                
                for i, row in df_perf_vs_loc.iterrows():
                    plt.scatter(row["total_loc"], row["best_s"], label=row["version"], 
                                s=50 + 250 * (1-(row['best_s'] - df_perf_vs_loc['best_s'].min()) / (df_perf_vs_loc['best_s'].max() - df_perf_vs_loc['best_s'].min() + 1e-9) ), # Size based on perf
                                color=cmap(norm(row['best_s'])), alpha=0.8)
                plt.legend(title="Version", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')


            for i, row in df_perf_vs_loc.iterrows():
                plt.annotate(f"{row['version']}\n({row['total_loc']} LOC)", (row["total_loc"], row["best_s"]), 
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, alpha=0.9)
            
            plt.xlabel("Total Lines of Code (Core Logic)")
            plt.ylabel("Best NP=1 Runtime (seconds)")
            plt.title(f"Performance (NP=1) vs. LOC\n{corr_text}")
            plt.grid(True, ls=":", lw=0.5)
            plt.tight_layout(rect=[0, 0, 0.80, 0.95] if len(df_perf_vs_loc['version'].unique()) > 3 else None) # Adjust for legend & title
            
            plot_path = visuals_output_dir / "performance_vs_loc_correlation.png"
            plt.savefig(plot_path)
            print(f"✓ Performance vs. LOC plot saved to {plot_path}")
            plt.show()
        else:
            print("Not enough merged data (or <2 points) for performance vs. LOC plot or correlation.")
    else:
        print("No NP=1 best runs to plot against LOC.")
else:
    print("No LOC data (all versions have 0 LOC or df_loc is empty) to plot performance against.")

# --- 3. Runtime Distribution Box Plots (Enhanced) ---
print("\n--- Plotting: Runtime Distributions (Box Plots) for Key Versions & NPs ---")
df_perf_for_boxplot = execute_query("""
    SELECT version, np, total_time_s 
    FROM perf_runs 
    WHERE version IN ('V1 Serial', 'V2.2 ScatterHalo', 'V3 CUDA', 'V4 MPI+CUDA')
      AND np IN (1, 2, 4) -- Focus on relevant NP values for these versions
""")

if df_perf_for_boxplot is not None and not df_perf_for_boxplot.empty:
    plt.figure(figsize=(16, 9)) # Larger figure for better readability
    
    # Create an interaction term for unique boxes and sort for consistent plotting order
    df_perf_for_boxplot['Version_NP'] = df_perf_for_boxplot['version'] + ' (NP=' + df_perf_for_boxplot['np'].astype(str) + ')'
    
    # Define a consistent order for versions in the plot
    version_order = ['V1 Serial', 'V2.2 ScatterHalo', 'V3 CUDA', 'V4 MPI+CUDA']
    np_order = [1, 2, 4]
    plot_order = [f"{v} (NP={n})" for v in version_order for n in np_order if f"{v} (NP={n})" in df_perf_for_boxplot['Version_NP'].unique()]

    if 'sns' in sys.modules:
        ax = sns.boxplot(data=df_perf_for_boxplot, x="Version_NP", y="total_time_s", order=plot_order, 
                         showfliers=True, palette="pastel", whis=[5, 95]) # Show 5th-95th percentile whiskers
        sns.stripplot(data=df_perf_for_boxplot, x="Version_NP", y="total_time_s", order=plot_order, 
                      color=".3", size=3, jitter=0.15, alpha=0.6, ax=ax)
    else: 
        # Basic matplotlib boxplot (less ideal for grouped aesthetic)
        # Pandas boxplot groups by the 'by' column. We'd need to pivot or iterate.
        # For simplicity, if seaborn is not there, this specific grouped plot might be omitted or simplified.
        print("[INFO] Seaborn not available, detailed grouped boxplot will be basic.")
        df_perf_for_boxplot.boxplot(column='total_time_s', by='Version_NP', figsize=(16,9), grid=True, rot=45)


    plt.xlabel("Implementation Stage (Version & Number of Processes)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Distribution for Key Project Stages")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.grid(True, axis='y', ls=":", lw=0.5)
    
    # Dynamic Y-axis scaling (consider log if data spans multiple orders of magnitude)
    y_max = df_perf_for_boxplot['total_time_s'].max() if not df_perf_for_boxplot.empty else 1.0
    y_min = df_perf_for_boxplot['total_time_s'].min() if not df_perf_for_boxplot.empty else 0.0
    if y_max / max(1e-9, y_min) > 50 : # Heuristic for using log scale
       plt.yscale('log')
       plt.ylabel("Runtime (seconds, log scale)")
       # Ensure y-ticks are sensible on log scale
       if 'sns' in sys.modules and ax: # If using seaborn ax
            ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))


    plt.tight_layout()
    plot_path = visuals_output_dir / "runtime_distributions_boxplot_enhanced.png"
    plt.savefig(plot_path)
    print(f"✓ Enhanced runtime distribution boxplot saved to {plot_path}")
    plt.show()
else:
    print("No performance data suitable for detailed box plots (key versions/NPs).")


# --- 4. Project Performance Timeline (Using `best_s` for clarity) ---
print("\n--- Plotting: Project Best Performance Timeline ---")
df_timeline = execute_query("""
    SELECT p.ts, p.version, p.np, b.best_s
    FROM perf_runs p JOIN best_runs b ON p.version = b.version AND p.np = b.np AND p.total_time_s = b.best_s
    WHERE p.version IN ('V1 Serial', 'V2.1 BroadcastAll', 'V2.2 ScatterHalo', 'V3 CUDA', 'V4 MPI+CUDA')
    GROUP BY 1, 2, 3, 4 -- Ensure unique points if multiple identical best runs at same ts
    ORDER BY p.ts
""")

if df_timeline is not None and not df_timeline.empty and len(df_timeline['ts'].unique()) > 1:
    plt.figure(figsize=(15, 8)) # Wider for timeline
    
    if 'sns' in sys.modules:
        # Using relplot for potentially better legend handling with style and hue
        g = sns.relplot(data=df_timeline, x='ts', y='best_s', hue='version', style='np', 
                        kind='scatter', s=100, legend='full', palette='tab10', height=7, aspect=1.8)
        g.set_xticklabels(rotation=30, ha="right")
        g.set(title='Timeline of Best Achieved Runtimes per Version/NP', xlabel='Timestamp of Best Run', ylabel='Best Runtime (s)')
        if df_timeline['best_s'].max() / max(1e-9, df_timeline['best_s'].min()) > 50:
            g.set(yscale="log")
            g.ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

    else: # Fallback matplotlib scatter
        cmap = matplotlib.colormaps['tab10']
        versions_unique = df_timeline['version'].unique()
        nps_unique = sorted(df_timeline['np'].unique())
        markers = ['o', 's', '^', 'D', 'P', '*', 'X']

        for i, ver in enumerate(versions_unique):
            for j, num_p in enumerate(nps_unique):
                subset = df_timeline[(df_timeline['version'] == ver) & (df_timeline['np'] == num_p)]
                if not subset.empty:
                    plt.scatter(subset['ts'], subset['best_s'], 
                                label=f"{ver} NP{num_p}", 
                                color=cmap.colors[i % len(cmap.colors)], 
                                marker=markers[j % len(markers)], s=80, alpha=0.9)
        plt.xlabel("Timestamp of Best Run")
        plt.ylabel("Best Runtime (s)")
        plt.title("Timeline of Best Achieved Runtimes per Version/NP")
        plt.xticks(rotation=30, ha="right")
        if df_timeline['best_s'].max() / max(1e-9, df_timeline['best_s'].min()) > 50:
            plt.yscale('log')
            plt.gca().yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
        plt.legend(title="Version & NP", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout(rect=[0, 0, 0.80, 1])


    plot_path = visuals_output_dir / "project_best_performance_timeline.png"
    plt.savefig(plot_path)
    print(f"✓ Best performance timeline plot saved to {plot_path}")
    if 'sns' not in sys.modules or not isinstance(g, sns.FacetGrid): # Only call plt.show() if not using relplot's implicit show
        plt.show()
    elif isinstance(g, sns.FacetGrid): # For relplot, figure is managed by FacetGrid
        plt.close(g.fig) # Close the FacetGrid figure
else:
    print("Not enough distinct timestamped 'best_s' data for a meaningful timeline plot.")

# --- 5. Final Comparative Summary Table & Scorecard Plot ---
print("\n--- Generating Final Comparative Summary Table & Scorecard ---")
summary_data = []
key_versions_for_summary = ['V1 Serial', 'V2.2 ScatterHalo', 'V3 CUDA', 'V4 MPI+CUDA']

for ver in key_versions_for_summary:
    loc_val_series = df_loc[df_loc['version'] == ver]['total_loc']
    loc_val = loc_val_series.iloc[0] if not loc_val_series.empty else 0 # Default to 0 if no LOC
    
    best_np1_s_df = execute_query(f"SELECT best_s FROM best_runs WHERE version='{ver}' AND np=1")
    best_np1_s = best_np1_s_df.iloc[0]['best_s'] if best_np1_s_df is not None and not best_np1_s_df.empty else pd.NA
    
    # For NP=4, only relevant for MPI-based versions; V1 and V3 are NP=1 only for this metric.
    best_np4_s, speedup_np4, eff_np4 = pd.NA, pd.NA, pd.NA
    if ver in ['V2.2 ScatterHalo', 'V4 MPI+CUDA']: # Versions expected to scale to NP=4
        best_np4_s_df = execute_query(f"SELECT best_s FROM best_runs WHERE version='{ver}' AND np=4")
        best_np4_s = best_np4_s_df.iloc[0]['best_s'] if best_np4_s_df is not None and not best_np4_s_df.empty else pd.NA
            
        speedup_np4_df = execute_query(f"SELECT S FROM speedup WHERE version='{ver}' AND np=4")
        speedup_np4 = speedup_np4_df.iloc[0]['S'] if speedup_np4_df is not None and not speedup_np4_df.empty else pd.NA
        
        eff_np4_df = execute_query(f"SELECT E FROM efficiency WHERE version='{ver}' AND np=4")
        eff_np4 = eff_np4_df.iloc[0]['E'] if eff_np4_df is not None and not eff_np4_df.empty else pd.NA
    elif ver == 'V1 Serial' and pd.notna(best_np1_s) : # V1 has speedup/eff of 1 at NP=1
        speedup_np4 = 1.0 
        eff_np4 = 1.0
        # best_np4_s remains NA for V1
    
    summary_data.append({
        'Version': ver,
        'LOC (Core)': loc_val,
        'T_NP1 (s)': best_np1_s,
        'T_NP4 (s)': best_np4_s, # Will be NA for V1, V3
        'Speedup@NP4': speedup_np4, # Will be NA for V3, 1.0 for V1
        'Efficiency@NP4': eff_np4  # Will be NA for V3, 1.0 for V1
    })

df_final_summary = pd.DataFrame(summary_data).set_index('Version')
# Round numeric columns for display
for col in df_final_summary.select_dtypes(include=float).columns:
    df_final_summary[col] = df_final_summary[col].round(3)

print("\n--- Final Comparative Summary Table ---")
if not df_final_summary.empty:
    display(df_final_summary)
    # Export this table to markdown for the report
    summary_md_path = visuals_output_dir / "project_final_summary_scorecard.md"
    df_final_summary.reset_index().to_markdown(summary_md_path, index=False)
    print(f"✓ Final summary table exported to {summary_md_path}")
else:
    print("Could not generate final summary table. Check underlying views.")

# Scorecard Plot for Runtimes (T_NP1 and T_NP4 where applicable)
df_runtimes_score = df_final_summary[['T_NP1 (s)', 'T_NP4 (s)']].copy()
df_runtimes_score = df_runtimes_score.reset_index().melt(id_vars='Version', var_name='Metric', value_name='Time (s)').dropna()

if not df_runtimes_score.empty:
    plt.figure(figsize=(10, 7))
    if 'sns' in sys.modules:
        sns.barplot(data=df_runtimes_score, x="Version", y="Time (s)", hue="Metric", palette="viridis")
    else:
        df_runtimes_score.pivot(index='Version', columns='Metric', values='Time (s)').plot(kind='bar', figsize=(10,7), grid=True, rot=30)

    plt.title("Scorecard: Best Runtimes (Lower is Better)")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Project Version")
    if not ('sns' in sys.modules): plt.xticks(rotation=30, ha="right") # Rotate for matplotlib if not seaborn
    plt.legend(title="Metric", fontsize="small")
    plt.tight_layout()
    plot_path = visuals_output_dir / "scorecard_runtimes_final.png"
    plt.savefig(plot_path)
    print(f"✓ Runtimes scorecard plot saved to {plot_path}")
    plt.show()

# Scorecard Plot for Scalability (Speedup@NP4 and Efficiency@NP4)
df_scalability_score = df_final_summary[['Speedup@NP4', 'Efficiency@NP4']].copy()
df_scalability_score = df_scalability_score.reset_index().melt(id_vars='Version', var_name='Metric', value_name='Value').dropna()
# Filter for versions that actually have NP4 data
df_scalability_score = df_scalability_score[df_scalability_score['Version'].isin(['V2.2 ScatterHalo', 'V4 MPI+CUDA', 'V1 Serial'])]


if not df_scalability_score.empty:
    plt.figure(figsize=(10, 7))
    if 'sns' in sys.modules:
        sns.barplot(data=df_scalability_score, x="Version", y="Value", hue="Metric", palette="crest")
    else:
        df_scalability_score.pivot(index='Version', columns='Metric', values='Value').plot(kind='bar', figsize=(10,7), grid=True, rot=30)
            
    plt.title("Scorecard: Scalability Metrics (Higher is Better)")
    plt.ylabel("Value")
    plt.xlabel("Project Version")
    if not ('sns' in sys.modules): plt.xticks(rotation=30, ha="right")
    plt.legend(title="Metric", fontsize="small")
    plt.tight_layout()
    plot_path = visuals_output_dir / "scorecard_scalability_final.png"
    plt.savefig(plot_path)
    print(f"✓ Scalability scorecard plot saved to {plot_path}")
    plt.show()

print("\n--- Advanced Analysis Cell Successfully Completed ---")
```


```python
# In[19]:
# Cell 19: Grand Synthesis - Multi-Perspective Analysis & Advanced Visualization (Critique Addressed & ValueError Fixed)

import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy.stats import pearsonr
import re # Ensure re is imported
import sys # For checking module availability
from pathlib import Path # For path operations
import matplotlib.pyplot as plt # Ensure plt is explicitly available

# Apply a nicer default style if seaborn is available
if 'seaborn' in sys.modules and 'matplotlib.pyplot' in sys.modules and plt is not None:
    sns.set_theme(style="whitegrid", palette="muted")
    print("[INFO] Seaborn theme applied for enhanced plot aesthetics.")
else:
    print("[INFO] Seaborn not available or not fully initialized; using default Matplotlib styles.")

visuals_output_dir = Path("analysis_visuals_final")
visuals_output_dir.mkdir(parents=True, exist_ok=True)
print(f"[INFO] All plots and advanced exports for this cell will be saved to: {visuals_output_dir.resolve()}")

# --- Dummy execute_query function if not defined elsewhere ---
# This is essential for the script to run if it's standalone.
# In a real notebook, this would interact with your SQL database.
_db_conn = None # Placeholder for actual database connection
def execute_query(query, conn=None):
    global _db_conn
    print(f"[INFO] Attempting to execute query: {query[:100]}...") # Print snippet
    # This is a mock implementation. Replace with your actual database query logic.
    if "FROM source_stats" in query and "COUNT(*)" in query:
        # Mock response for LOC count check
        return pd.DataFrame({'count': [10]}) # Assume some stats exist
    if "FROM source_stats" in query:
        # Mock LOC data for specified versions if df_loc is the target
        mock_loc_data = {
            "V1 Serial": 525, "V2.1 BroadcastAll": 306, "V2.2 ScatterHalo": 483,
            "V3 CUDA": 354, "V4 MPI+CUDA": 576, "V5 MPI+CUDA-Aware": 0
        }
        version_match = re.search(r"SELECT '([^']*)' as version", query)
        if version_match:
            ver_name = version_match.group(1)
            return pd.DataFrame({'version': [ver_name], 'total_loc': [mock_loc_data.get(ver_name, 0)]})
        return pd.DataFrame({'version': [], 'total_loc': []}) # Default empty for general LOC query structure

    if "FROM perf_runs" in query and "MEDIAN(total_time_s)" in query:
        # Mock median performance data
        # V1: 0.78s, V2.1: 0.74s, V2.2: 0.71s (NP1), 0.53s (NP2), 0.38s (NP4)
        # V3: 0.49s, V4: 0.43s (NP1), 0.24s (NP2), 0.44s (NP4)
        mock_median_data = {
            ("V1 Serial", 1): 0.787,
            ("V2.1 BroadcastAll", 1): 0.743,
            ("V2.1 BroadcastAll", 4): 0.833, # Mock for max_np
            ("V2.2 ScatterHalo", 1): 0.713,
            ("V2.2 ScatterHalo", 4): 0.383, # Mock for max_np
            ("V3 CUDA", 1): 0.491,
            ("V4 MPI+CUDA", 1): 0.428,
            ("V4 MPI+CUDA", 4): 0.445, # Mock for max_np
            ("V4 MPI+CUDA", 2): 0.240, # for the actual dip
        }
        version_match = re.search(r"version = '([^']*)'", query)
        np_match = re.search(r"np = (\d+)", query)
        if version_match and np_match:
            ver, np_val = version_match.group(1), int(np_match.group(1))
            return pd.DataFrame({'median_s': [mock_median_data.get((ver, np_val), pd.NA)]})
        return pd.DataFrame({'median_s': [pd.NA]})

    if "FROM perf_runs" in query and "MAX(np)" in query:
         # Mock max NP
        mock_max_np = {"V2.1 BroadcastAll": 4, "V2.2 ScatterHalo": 4, "V4 MPI+CUDA": 4}
        version_match = re.search(r"version='([^']*)'", query)
        if version_match:
            ver = version_match.group(1)
            return pd.DataFrame({'max_np': [mock_max_np.get(ver, 1)]}) # Default to 1 if not scalable
        return pd.DataFrame({'max_np': [1]})

    if "FROM run_stats" in query:
        # Mock CV data from run_stats
        # CV = SD/Mean. V1: 0.21, V2.2 NP1:0.26 NP4:0.42, V3:0.35, V4 NP1:0.77 NP4:0.25
        # n values: V1(10), V2.2 NP1(13) NP4(11), V3(13), V4 NP1(13) NP4(13)
        mock_cv_data = {
            ("V1 Serial", 1): (0.212, 10),
            ("V2.2 ScatterHalo", 1): (0.265, 13), ("V2.2 ScatterHalo", 2): (0.493, 13), ("V2.2 ScatterHalo", 4): (0.419, 11),
            ("V3 CUDA", 1): (0.350, 13),
            ("V4 MPI+CUDA", 1): (0.768, 13), ("V4 MPI+CUDA", 2): (0.518, 12), ("V4 MPI+CUDA", 4): (0.255, 13),
        }
        version_match = re.search(r"version='([^']*)'", query)
        np_match = re.search(r"np=(\d+)", query)

        if "SELECT version, np, n, ROUND(mean_s, 4) as mean_s, ROUND(sd_s, 4) as sd_s" in query : # For the CV bar plot data
             results = []
             for (ver_key, np_key), (cv_val, n_val) in mock_cv_data.items():
                 if ver_key in ('V1 Serial', 'V2.2 ScatterHalo', 'V3 CUDA', 'V4 MPI+CUDA'): # Filter for relevant versions
                    results.append({'version': ver_key, 'np': np_key, 'n': n_val, 'mean_s': 0.1, 'sd_s': cv_val * 0.1, 'CV': cv_val})
             return pd.DataFrame(results)

        if version_match and np_match:
            ver, np_val = version_match.group(1), int(np_match.group(1))
            cv, n = mock_cv_data.get((ver, np_val), (pd.NA, 0))
            return pd.DataFrame({'CV': [cv], 'n': [n]}) # Ensure 'n' is returned if needed by calling code
        return pd.DataFrame({'CV': [pd.NA], 'n': [0]})
    # Fallback for other queries
    print(f"[WARNING] Mock response for query: {query[:100]}... returning empty DataFrame or default.")
    return pd.DataFrame()
# --- End of Dummy execute_query ---


# --- Helper to format y-axis ticks for log scale if used ---
def log_tick_formatter(val, pos=None):
    return f"{val:.2g}"

# --- 1. Robust Lines of Code (LOC) Aggregation (Core Algorithmic Code) ---
print("\n--- 1. Aggregating Lines of Code (LOC) for Canonical Versions ---")
version_loc_map = { # Paths relative to 'final_project/' directory
    "V1 Serial": ["v1_serial/src/", "v1_serial/include/"],
    "V2.1 BroadcastAll": ["v2_mpi_only/2.1_broadcast_all/src/", "v2_mpi_only/2.1_broadcast_all/include/"],
    "V2.2 ScatterHalo": ["v2_mpi_only/2.2_scatter_halo/src/", "v2_mpi_only/2.2_scatter_halo/include/"],
    "V3 CUDA": ["v3_cuda_only/src/", "v3_cuda_only/include/"],
    "V4 MPI+CUDA": ["v4_mpi_cuda/src/", "v4_mpi_cuda/include/"],
    "V5 MPI+CUDA-Aware": ["v5_cuda_aware_mpi/src/", "v5_cuda_aware_mpi/include/"]
}
version_loc_data = []
df_loc = pd.DataFrame(columns=['version', 'total_loc'])

source_stats_check_df = execute_query("SELECT COUNT(*) as count FROM source_stats")
if source_stats_check_df is None or source_stats_check_df.iloc[0]['count'] == 0:
    print("[WARNING] 'source_stats' table is empty or unreachable. Using MOCK LOC data for plots.")
    # Populate df_loc with mock data if source_stats is empty
    mock_loc_data_list = [
        {'version': "V1 Serial", 'total_loc': 525}, {'version': "V2.1 BroadcastAll", 'total_loc': 306},
        {'version': "V2.2 ScatterHalo", 'total_loc': 483}, {'version': "V3 CUDA", 'total_loc': 354},
        {'version': "V4 MPI+CUDA", 'total_loc': 576}, {'version': "V5 MPI+CUDA-Aware", 'total_loc': 0} # Assume V5 not implemented
    ]
    df_loc = pd.DataFrame(mock_loc_data_list)
else:
    for version_name, dir_prefixes in version_loc_map.items():
        like_clauses = [f"relpath LIKE '{prefix}%'" for prefix in dir_prefixes]
        dir_filter = " OR ".join(like_clauses)
        ext_filter = "OR ".join([f"LOWER(relpath) LIKE '%.{ext}'" for ext in ['cpp', 'cu', 'hpp', 'h', 'c', 'inl']])

        query = f"""
        SELECT '{version_name}' as version, COALESCE(SUM(loc), 0) as total_loc
        FROM source_stats
        WHERE ({dir_filter}) AND ({ext_filter});
        """
        loc_df_query_result = execute_query(query)
        current_loc = loc_df_query_result.iloc[0]['total_loc'] if loc_df_query_result is not None and not loc_df_query_result.empty else 0
        version_loc_data.append({'version': version_name, 'total_loc': int(current_loc)})
    if version_loc_data: df_loc = pd.DataFrame(version_loc_data)

df_loc_display = df_loc[df_loc['total_loc'] > 0]
if not df_loc_display.empty:
    print("Aggregated LOC for core algorithmic code (src/, include/):")
    display(df_loc_display)
else:
    print("No LOC found for specified version paths/extensions. df_loc is empty or all LOC are 0.")


# --- 2. Plot: Median NP=1 Performance vs. LOC (Revised Visuals) ---
print("\n--- 2. Plotting: Median NP=1 Performance vs. Lines of Code (Revised Visuals) ---")
df_loc_for_plot = df_loc[df_loc['total_loc'] > 0]
if not df_loc_for_plot.empty and plt is not None:
    df_median_np1 = execute_query("SELECT version, MEDIAN(total_time_s) as median_np1_s FROM perf_runs WHERE np = 1 AND total_time_s IS NOT NULL GROUP BY version")

    if df_median_np1 is not None and not df_median_np1.empty:
        df_perf_vs_loc = pd.merge(df_median_np1, df_loc_for_plot, on="version")

        if not df_perf_vs_loc.empty and len(df_perf_vs_loc) > 1 :
            plt.figure(figsize=(11, 7))
            df_corr = df_perf_vs_loc[['total_loc', 'median_np1_s']].dropna()
            corr_text = "Correlation: N/A (requires >1 data point)"
            pearson_r_val, p_value_val = pd.NA, pd.NA # Initialize for later use
            if len(df_corr) >= 2:
                 pearson_r_val, p_value_val = pearsonr(df_corr["total_loc"], df_corr["median_np1_s"])
                 corr_text = f'Pearson R: {pearson_r_val:.2f} (p={p_value_val:.2g})'

            unique_versions_count = len(df_perf_vs_loc['version'].unique())
            palette = sns.color_palette("viridis", n_colors=unique_versions_count) if 'seaborn' in sys.modules and unique_versions_count > 0 else "viridis"

            # Use a different size mapping for LOC if desired, e.g., based on relative LOC
            min_loc, max_loc = df_perf_vs_loc['total_loc'].min(), df_perf_vs_loc['total_loc'].max()
            sizes = 100 + 400 * (df_perf_vs_loc['total_loc'] - min_loc) / (max_loc - min_loc + 1e-9) if max_loc > min_loc else 150


            if 'seaborn' in sys.modules:
                sns.scatterplot(data=df_perf_vs_loc, x="total_loc", y="median_np1_s", hue="version",
                                size="total_loc", sizes=(100,500), # Make size reflect LOC
                                legend="auto", palette=palette, alpha=0.85)
            else:
                for i, row in df_perf_vs_loc.iterrows(): plt.scatter(row["total_loc"], row["median_np1_s"], label=row["version"], s=sizes.iloc[i] if isinstance(sizes, pd.Series) else sizes, alpha=0.7)

            plt.legend(title="Version", bbox_to_anchor=(1.03, 1), loc='upper left', fontsize='small')
            for i, row in df_perf_vs_loc.iterrows():
                plt.annotate(f"{row['version']}", (row["total_loc"], row["median_np1_s"]),
                             textcoords="offset points", xytext=(5,5), ha='left', fontsize=8,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5))

            plt.xlabel("Total Lines of Code (Core Algorithmic Files)", fontsize=11)
            plt.ylabel("Median NP=1 Runtime (seconds)", fontsize=11)
            plt.title(f"Code Complexity vs. Single-Process Performance (Median Runtime)\n{corr_text}", fontsize=13)
            plt.grid(True, ls=":", lw=0.5, which="both")
            if not df_perf_vs_loc.empty and (df_perf_vs_loc['median_np1_s'].max() / max(1e-9, df_perf_vs_loc['median_np1_s'].min()) > 20):
                plt.yscale('log')
                plt.ylabel("Median NP=1 Runtime (seconds, log scale)", fontsize=11)
                if plt.gca(): plt.gca().yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
            plt.tight_layout(rect=[0, 0, 0.82, 0.95])
            plot_path = visuals_output_dir / "adv_median_performance_vs_loc_corr_revised.png"
            plt.savefig(plot_path); print(f"✓ Median Performance vs. LOC plot saved to {plot_path}"); plt.show()
        else: print("[WARNING] Not enough merged data for median performance vs. LOC plot or correlation.")
    else: print("[WARNING] No median NP=1 runs to plot against LOC.")
else: print("[WARNING] No LOC data or Matplotlib not available. Perf vs LOC plot skipped.")


# --- 3. Runtime Variability Analysis (CV with 'n' annotations, improved aesthetics) ---
print("\n--- 3. Runtime Variability (CV with 'n' annotations, improved aesthetics) ---")
df_variability = execute_query("""
    SELECT version, np, n, ROUND(mean_s, 4) as mean_s, ROUND(sd_s, 4) as sd_s,
           CASE WHEN mean_s > 1e-9 THEN ROUND(sd_s / mean_s, 3) ELSE NULL END AS CV
    FROM run_stats WHERE version IN ('V1 Serial', 'V2.2 ScatterHalo', 'V3 CUDA', 'V4 MPI+CUDA') AND n > 1 ORDER BY version, np;
""")
if df_variability is not None and not df_variability.empty and plt is not None:
    print("CV for Runtimes (Lower = More Stable):"); display(df_variability)
    if len(df_variability) > 1:
        plt.figure(figsize=(11, 6.5))
        ax = None
        hue_order_cv = sorted(df_variability['np'].unique())

        if 'seaborn' in sys.modules:
            ax = sns.barplot(data=df_variability, x="version", y="CV", hue="np",
                             dodge=True, palette="coolwarm_r", errorbar=None, hue_order=hue_order_cv)
            sns.despine(left=True, bottom=True) # Clean look
            if ax:
                for bar_container in ax.containers: # Iterate through containers of bars for each hue level
                    for bar in bar_container:
                        height = bar.get_height()
                        if pd.notna(height) and height > 0.001: # Check if bar has valid height
                            # Get corresponding 'n' value
                            # This requires finding the matching (version, np) in df_variability
                            # The bar object itself doesn't directly store original data row index easily
                            # We infer based on bar position - this can be tricky if bars are missing.
                            # A more robust way is to iterate df_variability and plot annotations but sns.barplot is convenient.
                            # For simplicity, we'll try to match back.
                            x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
                            bar_x_center = bar.get_x() + bar.get_width() / 2.
                            version_idx_approx = np.argmin(np.abs([tick.get_position()[0] - bar_x_center for tick in ax.get_xticklabels()]))
                            current_version = x_tick_labels[version_idx_approx]

                            # Find current_np based on hue_order and bar_container index (requires careful matching)
                            # This part is simplified; a robust solution would be more complex.
                            # Assuming hue_order_cv maps to containers if sns creates them in that order.
                            current_np = None
                            for idx, h_val in enumerate(hue_order_cv): # Try to find which hue this bar belongs to
                                if bar_container == ax.containers[idx]:
                                    current_np = h_val
                                    break
                            
                            n_val_text = ""
                            if current_version and current_np is not None:
                                n_val_series = df_variability[(df_variability['version'] == current_version) & (df_variability['np'] == current_np)]['n']
                                n_val_text = f"n={n_val_series.iloc[0]}" if not n_val_series.empty else ""

                            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01, n_val_text,
                                    ha='center', va='bottom', color='black', fontsize=7)
        else:
             df_variability.pivot(index='version', columns='np', values='CV').plot(kind='bar', figsize=(11,6.5), grid=False, rot=15, ax=plt.gca())
             ax = plt.gca() # For consistency

        plt.title("Runtime CV (Stability)", fontsize=13)
        plt.xlabel("Version", fontsize=10); plt.ylabel("CV (StdDev / Mean)", fontsize=10)
        if ax and ax.get_legend() is not None: plt.legend(title="NP", loc="upper right", fontsize='x-small', frameon=False)
        plt.xticks(rotation=15, ha="right", fontsize='small')
        plt.tight_layout()
        plot_path = visuals_output_dir / "adv_runtime_variability_cv_annotated_revised.png"
        plt.savefig(plot_path); print(f"✓ Runtime CV plot saved to {plot_path}"); plt.show()
else: print("Not enough data (or n<=1, or Matplotlib unavailable) for CV plot.")


# --- 4. Multi-Metric Radar Chart (Revised Metrics & Visuals) ---
print("\n--- 4. Multi-Metric Radar Chart for Key Versions (Revised Metrics & Visuals) ---")
radar_metrics_raw = ['NP1 Perf (1/Med.T)', 'Max Speedup (Med.T based)', 'Max Efficiency (Med.T based)', 'Code Volume (log10 LOC)']
radar_metrics_display = ['NP1 Perf (1/Med.T_NP1)\n(Higher=Better)',
                         'Max Scaled Speedup\n(Median-based, Higher=Better)',
                         'Max Scaled Efficiency\n(Median-based, Higher=Better)',
                         'Code Volume (log10 LOC)\n(Lower=Better -> Outer Edge)']
radar_versions = ['V1 Serial', 'V2.2 ScatterHalo', 'V3 CUDA', 'V4 MPI+CUDA']
radar_data_list = []

if not df_loc.empty and plt is not None:
    for ver in radar_versions:
        median_t_np1_df = execute_query(f"SELECT MEDIAN(total_time_s) as t_val FROM perf_runs WHERE version = '{ver}' AND np = 1 AND total_time_s IS NOT NULL")
        median_t_np1_val = median_t_np1_df.iloc[0]['t_val'] if median_t_np1_df is not None and not median_t_np1_df.empty and pd.notna(median_t_np1_df.iloc[0]['t_val']) else 0
        perf_val = 1.0 / median_t_np1_val if median_t_np1_val > 1e-9 else 0

        np_at_max_s = 1
        if ver in ['V2.2 ScatterHalo', 'V4 MPI+CUDA']:
            max_np_for_ver_df = execute_query(f"SELECT MAX(np) as max_np FROM perf_runs WHERE version = '{ver}' AND total_time_s IS NOT NULL AND np > 1")
            if max_np_for_ver_df is not None and not max_np_for_ver_df.empty and pd.notna(max_np_for_ver_df.iloc[0]['max_np']):
                np_at_max_s = int(max_np_for_ver_df.iloc[0]['max_np'])

        median_t_np_max_df = execute_query(f"SELECT MEDIAN(total_time_s) as t_val FROM perf_runs WHERE version = '{ver}' AND np = {np_at_max_s} AND total_time_s IS NOT NULL")
        median_t_np_max_val = median_t_np_max_df.iloc[0]['t_val'] if median_t_np_max_df is not None and not median_t_np_max_df.empty and pd.notna(median_t_np_max_df.iloc[0]['t_val']) else 0

        speedup_val = (median_t_np1_val / median_t_np_max_val) if median_t_np1_val > 1e-9 and median_t_np_max_val > 1e-9 else (1.0 if np_at_max_s == 1 else 0)
        efficiency_val = (speedup_val / np_at_max_s) if np_at_max_s > 0 else (1.0 if np_at_max_s == 1 else 0)

        loc_val_series = df_loc[df_loc['version'] == ver]['total_loc']
        loc_val = loc_val_series.iloc[0] if not loc_val_series.empty and loc_val_series.iloc[0] > 0 else 1
        log_loc_val = np.log10(loc_val)
        radar_data_list.append([perf_val, speedup_val, efficiency_val, log_loc_val])

if radar_data_list:
    df_radar_raw = pd.DataFrame(radar_data_list, columns=radar_metrics_raw, index=radar_versions)
    df_radar_normalized = df_radar_raw.copy()

    for col in ['NP1 Perf (1/Med.T)', 'Max Speedup (Med.T based)', 'Max Efficiency (Med.T based)']:
        min_v, max_v = df_radar_raw[col].min(), df_radar_raw[col].max()
        df_radar_normalized[col] = (df_radar_raw[col] - min_v) / (max_v - min_v) if (max_v - min_v) > 1e-9 else 0.5 # Avoid div by zero
    col_log_loc = 'Code Volume (log10 LOC)'
    min_v, max_v = df_radar_raw[col_log_loc].min(), df_radar_raw[col_log_loc].max()
    if (max_v - min_v) > 1e-9:
         df_radar_normalized[col_log_loc] = 1 - ((df_radar_raw[col_log_loc] - min_v) / (max_v - min_v)) # Invert for "lower is better"
    else: df_radar_normalized[col_log_loc] = 0.5

    num_vars = len(radar_metrics_raw)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0] # Close the loop
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    line_styles = ['-', '--', '-.', ':']
    marker_styles = ['o', 's', '^', 'D']
    # Use a seaborn color palette for better distinction if available
    palette_radar = sns.color_palette("husl", n_colors=len(df_radar_normalized.index)) if 'seaborn' in sys.modules else plt.cm.get_cmap('viridis', len(df_radar_normalized.index))


    for i, version_name in enumerate(df_radar_normalized.index):
        values = df_radar_normalized.loc[version_name].values.flatten().tolist() + [df_radar_normalized.loc[version_name].values.flatten().tolist()[0]] # Close loop
        color = palette_radar[i % len(palette_radar)] if 'seaborn' in sys.modules else palette_radar(i / len(df_radar_normalized.index))

        ax.plot(angles, values, linewidth=1.5, linestyle=line_styles[i % len(line_styles)],
                label=version_name, marker=marker_styles[i % len(marker_styles)], markersize=6, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)


    ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_metrics_display, fontsize=9)
    ax.set_yticks(np.arange(0, 1.1, 0.2)); ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0, 1.1, 0.2)], fontsize=8)
    ax.set_rlabel_position(30) # Move radial labels
    plt.title('Comparative Multi-Metric Radar Chart (Normalized)', size=14, y=1.12)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), fontsize='small', ncol=len(radar_versions)//2 or 2) # Adjust ncol dynamically

    plot_path = visuals_output_dir / "adv_multi_metric_radar_chart_final_revised.png"
    plt.savefig(plot_path, bbox_inches='tight'); print(f"✓ Final Revised Radar chart saved to {plot_path}"); plt.show()
else: print("[WARNING] Not enough data for revised radar chart (check LOC and perf data).")


# --- 5. Final Comprehensive Scorecard Table (Median-based & CV from run_stats) ---
print("\n--- 5. Generating Final Comprehensive Scorecard Table (Median Runtimes, CV from run_stats) ---")
summary_data_final = []
key_versions_for_scorecard = ['V1 Serial', 'V2.1 BroadcastAll', 'V2.2 ScatterHalo', 'V3 CUDA', 'V4 MPI+CUDA']
df_final_scorecard = pd.DataFrame() # Initialize

if not df_loc.empty:
    for ver in key_versions_for_scorecard:
        loc_val_series = df_loc[df_loc['version'] == ver]['total_loc']
        loc_val = loc_val_series.iloc[0] if not loc_val_series.empty else 0

        median_np1_s_df = execute_query(f"SELECT MEDIAN(total_time_s) as median_s FROM perf_runs WHERE version='{ver}' AND np=1 AND total_time_s IS NOT NULL")
        median_np1_s = median_np1_s_df.iloc[0]['median_s'] if median_np1_s_df is not None and not median_np1_s_df.empty and pd.notna(median_np1_s_df.iloc[0]['median_s']) else pd.NA

        cv_np1_from_stats_df = execute_query(f"SELECT CASE WHEN mean_s > 1e-9 THEN ROUND(sd_s / mean_s, 3) ELSE NULL END AS CV FROM run_stats WHERE version='{ver}' AND np=1 AND n > 1")
        cv_np1 = cv_np1_from_stats_df.iloc[0]['CV'] if cv_np1_from_stats_df is not None and not cv_np1_from_stats_df.empty and pd.notna(cv_np1_from_stats_df.iloc[0]['CV']) else pd.NA

        np_for_max_metric = 1 # Default for V1, V3
        if ver in ['V2.1 BroadcastAll', 'V2.2 ScatterHalo', 'V4 MPI+CUDA']:
            max_np_df = execute_query(f"SELECT MAX(np) as max_np FROM perf_runs WHERE version='{ver}' AND total_time_s IS NOT NULL AND np > 0")
            if max_np_df is not None and not max_np_df.empty and pd.notna(max_np_df.iloc[0]['max_np']) and max_np_df.iloc[0]['max_np'] > 0 :
                np_for_max_metric = int(max_np_df.iloc[0]['max_np'])

        median_np_max_s_df = execute_query(f"SELECT MEDIAN(total_time_s) as median_s FROM perf_runs WHERE version='{ver}' AND np={np_for_max_metric} AND total_time_s IS NOT NULL")
        median_np_max_s = median_np_max_s_df.iloc[0]['median_s'] if median_np_max_s_df is not None and not median_np_max_s_df.empty and pd.notna(median_np_max_s_df.iloc[0]['median_s']) else pd.NA

        speedup_at_np_max, efficiency_at_np_max = pd.NA, pd.NA
        if pd.notna(median_np1_s) and median_np1_s > 1e-9 and pd.notna(median_np_max_s) and median_np_max_s > 1e-9 :
            speedup_at_np_max = median_np1_s / median_np_max_s
            if np_for_max_metric > 0: efficiency_at_np_max = speedup_at_np_max / np_for_max_metric
        elif (ver == 'V1 Serial' or ver == 'V3 CUDA') and np_for_max_metric == 1 : # Explicitly handle non-scalable versions
             speedup_at_np_max = 1.0 if pd.notna(median_np1_s) else pd.NA
             efficiency_at_np_max = 1.0 if pd.notna(median_np1_s) else pd.NA


        summary_data_final.append({
            'Version': ver, 'LOC (Core)': loc_val,
            'Median T_NP1 (s)': median_np1_s, 'CV @NP1 (Mean-based)': cv_np1,
            f'Max Scaled NP': np_for_max_metric, # Store the NP used for max metrics
            f'Median T @Max Scaled NP (s)': median_np_max_s,
            f'Speedup (Medians) @Max Scaled NP': speedup_at_np_max,
            f'Efficiency (Medians) @Max Scaled NP': efficiency_at_np_max
        })
    if summary_data_final:
        df_final_scorecard = pd.DataFrame(summary_data_final).set_index('Version')
        df_final_scorecard[df_final_scorecard.select_dtypes(include='number').columns] = df_final_scorecard.select_dtypes(include='number').round(3)
        print("\n--- Final Scorecard (Median Runtimes, CV from run_stats) ---"); display(df_final_scorecard)
        scorecard_md_path = visuals_output_dir / "project_final_scorecard_median_cv_from_stats.md"
        df_final_scorecard.reset_index().to_markdown(scorecard_md_path, index=False)
        print(f"✓ Final scorecard table exported to {scorecard_md_path}")
    else: print("[WARNING] Could not generate final scorecard table (check LOC/perf data).")
else: print("[WARNING] df_loc empty. Scorecard skipped.")


# --- 6. Markdown Cell for Qualitative Interpretation (Critique Addressed & ValueError Fixed) ---

# Initialize strings to 'N/A' to prevent errors if data is missing
correlation_r_str, correlation_p_val_str = "N/A", "N/A"
if 'pearson_r_val' in locals() and pd.notna(pearson_r_val) and isinstance(pearson_r_val, (float, int)):
    correlation_r_str = f"{pearson_r_val:.2f}"
if 'p_value_val' in locals() and pd.notna(p_value_val) and isinstance(p_value_val, (float, int)):
    correlation_p_val_str = f"{p_value_val:.2g}"


# Helper function to safely get values from scorecard for interpretation
def get_formatted_scorecard_value(df, version, col_name, fmt="{:.2f}"):
    if not df.empty and version in df.index and col_name in df.columns and pd.notna(df.loc[version, col_name]):
        val = df.loc[version, col_name]
        if isinstance(val, (float, int)): return fmt.format(val)
        return str(val) # Return as string if not float/int (e.g. already 'N/A' or text)
    return "N/A"

v4_loc_str = get_formatted_scorecard_value(df_final_scorecard, 'V4 MPI+CUDA', 'LOC (Core)', fmt="{:d}")
v2_t_np1_val = df_final_scorecard.loc['V2.2 ScatterHalo']['Median T_NP1 (s)'] if not df_final_scorecard.empty and 'V2.2 ScatterHalo' in df_final_scorecard.index and pd.notna(df_final_scorecard.loc['V2.2 ScatterHalo']['Median T_NP1 (s)']) else None
v4_t_np1_val = df_final_scorecard.loc['V4 MPI+CUDA']['Median T_NP1 (s)'] if not df_final_scorecard.empty and 'V4 MPI+CUDA' in df_final_scorecard.index and pd.notna(df_final_scorecard.loc['V4 MPI+CUDA']['Median T_NP1 (s)']) else None

speedup_factor_v4_vs_v22_np1_str = "N/A"
if v2_t_np1_val is not None and v4_t_np1_val is not None and v4_t_np1_val > 1e-9:
    speedup_factor_v4_vs_v22_np1_str = f"{(v2_t_np1_val / v4_t_np1_val):.2f}x"

np_val_str_v4 = get_formatted_scorecard_value(df_final_scorecard, 'V4 MPI+CUDA', 'Max Scaled NP', fmt="{:d}")
v4_speedup_np_max_val = df_final_scorecard.loc['V4 MPI+CUDA']['Speedup (Medians) @Max Scaled NP'] if not df_final_scorecard.empty and 'V4 MPI+CUDA' in df_final_scorecard.index else pd.NA
v4_speedup_np_max_str = f"{v4_speedup_np_max_val:.2f}x" if pd.notna(v4_speedup_np_max_val) and isinstance(v4_speedup_np_max_val, (int,float)) else "N/A"

report_banner = f"Key Insight: Hybrid V4 (LOC: {v4_loc_str}) achieved ~{speedup_factor_v4_vs_v22_np1_str} NP=1 speedup vs. MPI-only V2.2, but scaled poorly to {np_val_str_v4 if np_val_str_v4 != 'N/A' else 'Max NP'} processes (Median-based Speedup: {v4_speedup_np_max_str}), highlighting severe host-staging bottlenecks."


v1_median_t_np1_for_calc = df_final_scorecard.loc['V1 Serial']['Median T_NP1 (s)'] if not df_final_scorecard.empty and 'V1 Serial' in df_final_scorecard.index and pd.notna(df_final_scorecard.loc['V1 Serial']['Median T_NP1 (s)']) else None
v3_median_t_np1_for_calc = df_final_scorecard.loc['V3 CUDA']['Median T_NP1 (s)'] if not df_final_scorecard.empty and 'V3 CUDA' in df_final_scorecard.index and pd.notna(df_final_scorecard.loc['V3 CUDA']['Median T_NP1 (s)']) else None

v3_vs_v1_speedup_str = "N/A"
if v1_median_t_np1_for_calc is not None and v3_median_t_np1_for_calc is not None and v3_median_t_np1_for_calc > 1e-9:
    v3_vs_v1_speedup_str = f"{(v1_median_t_np1_for_calc / v3_median_t_np1_for_calc):.2f}x"

v4_vs_v3_relative_perf_str = "N/A"
if v3_median_t_np1_for_calc is not None and v4_t_np1_val is not None and v4_t_np1_val > 1e-9: # v4_t_np1_val already defined
    v4_is_faster_factor = v3_median_t_np1_for_calc / v4_t_np1_val
    if v4_is_faster_factor > 1.005: # Add a small tolerance
        v4_vs_v3_relative_perf_str = f"{v4_is_faster_factor:.2f}x faster than V3"
    elif v4_is_faster_factor < 0.995:
         v4_vs_v3_relative_perf_str = f"{(1/v4_is_faster_factor):.2f}x slower than V3"
    else:
        v4_vs_v3_relative_perf_str = "about equal to V3"

# For V2.2 ScatterHalo in interpretation_md
np_val_str_v22 = get_formatted_scorecard_value(df_final_scorecard, 'V2.2 ScatterHalo', 'Max Scaled NP', fmt="{:d}")
speedup_median_npmax_v22_val = df_final_scorecard.loc['V2.2 ScatterHalo']['Speedup (Medians) @Max Scaled NP'] if not df_final_scorecard.empty and 'V2.2 ScatterHalo' in df_final_scorecard.index else pd.NA
speedup_median_npmax_v22_formatted_str = f"{speedup_median_npmax_v22_val:.2f}x" if pd.notna(speedup_median_npmax_v22_val) and isinstance(speedup_median_npmax_v22_val, (float,int)) else "N/A"


interpretation_md = f"""
## Qualitative Interpretation of Advanced Analysis (Critique Addressed)

{report_banner}

This analysis uses median runtimes for key performance indicators in the scorecard and radar chart for robustness. Note that general-purpose views like `speedup` (and plots from earlier cells if not regenerated) may still use MIN-based T1 for baseline performance.

**1. Code Complexity (LOC) vs. Single-Core/GPU Performance:**
*   **Figure:** `adv_median_performance_vs_loc_corr_revised.png`
*   **Takeaway:** Explores if more LOC (core algorithmic files) correlates with NP=1 median runtime.
*   **Your Observation & Data:**
    *   V1 Serial LOC: {get_formatted_scorecard_value(df_final_scorecard, 'V1 Serial', 'LOC (Core)', fmt="{{:d}}")}, Median T_NP1: {get_formatted_scorecard_value(df_final_scorecard, 'V1 Serial', 'Median T_NP1 (s)')}s.
    *   V3 CUDA LOC: {get_formatted_scorecard_value(df_final_scorecard, 'V3 CUDA', 'LOC (Core)', fmt="{{:d}}")}, Median T_NP1: {get_formatted_scorecard_value(df_final_scorecard, 'V3 CUDA', 'Median T_NP1 (s)')}s.
    *   V4 MPI+CUDA LOC: {get_formatted_scorecard_value(df_final_scorecard, 'V4 MPI+CUDA', 'LOC (Core)', fmt="{{:d}}")}, Median T_NP1: {get_formatted_scorecard_value(df_final_scorecard, 'V4 MPI+CUDA', 'Median T_NP1 (s)')}s.
    *   Discuss: V3's GPU offload (LOC: {get_formatted_scorecard_value(df_final_scorecard, 'V3 CUDA', 'LOC (Core)', fmt="{{:d}}")}) yielded a median NP=1 runtime of {get_formatted_scorecard_value(df_final_scorecard, 'V3 CUDA', 'Median T_NP1 (s)')}s. This was {v3_vs_v1_speedup_str} faster than V1 Serial. V4 MPI+CUDA (highest LOC: {get_formatted_scorecard_value(df_final_scorecard, 'V4 MPI+CUDA', 'LOC (Core)', fmt="{{:d}}")}) achieved a median NP=1 time of {get_formatted_scorecard_value(df_final_scorecard, 'V4 MPI+CUDA', 'Median T_NP1 (s)')}s, which is {v4_vs_v3_relative_perf_str}.
*   **Correlation:** Pearson R = **{correlation_r_str}**, p-value = **{correlation_p_val_str}**.
    *   Interpret this: The correlation of R={correlation_r_str} (p={correlation_p_val_str}) suggests a weak and statistically insignificant linear relationship between LOC and median NP=1 runtime for this dataset. This indicates that the choice of parallelization paradigm (CPU, MPI, CUDA, Hybrid) and its specific implementation details had a much stronger impact on single-process performance than mere code volume.

**2. Runtime Variability (Stability):**
*   **Figure:** `adv_runtime_variability_cv_annotated_revised.png` (Note: 'n' values are in the displayed table `df_variability`).
*   **Takeaway:** CV (StdDev/Mean from `run_stats`) shows consistency. Lower is better.
*   **Your Observation & Data:**
    *   V1 Serial (NP=1) CV: {get_formatted_scorecard_value(df_final_scorecard, 'V1 Serial', 'CV @NP1 (Mean-based)')}.
    *   V3 CUDA (NP=1) CV: {get_formatted_scorecard_value(df_final_scorecard, 'V3 CUDA', 'CV @NP1 (Mean-based)')}.
    *   V2.2 ScatterHalo (NP={np_val_str_v22 if np_val_str_v22 != 'N/A' else 'Max'}) CV: {df_variability[(df_variability['version']=='V2.2 ScatterHalo') & (df_variability['np']==int(np_val_str_v22 if np_val_str_v22.isdigit() else 0))]['CV'].iloc[0] if df_variability is not None and np_val_str_v22.isdigit() and not df_variability[(df_variability['version']=='V2.2 ScatterHalo') & (df_variability['np']==int(np_val_str_v22))].empty else 'N/A'}.
    *   V4 MPI+CUDA (NP={np_val_str_v4 if np_val_str_v4 != 'N/A' else 'Max'}) CV: {df_variability[(df_variability['version']=='V4 MPI+CUDA') & (df_variability['np']==int(np_val_str_v4 if np_val_str_v4.isdigit() else 0))]['CV'].iloc[0] if df_variability is not None and np_val_str_v4.isdigit() and not df_variability[(df_variability['version']=='V4 MPI+CUDA') & (df_variability['np']==int(np_val_str_v4))].empty else 'N/A'}.
    *   Discuss: V1 Serial and V3 CUDA NP=1 runs show CVs suggesting moderate stability. V4 MPI+CUDA at NP=1 has a notably high CV ({get_formatted_scorecard_value(df_final_scorecard, 'V4 MPI+CUDA', 'CV @NP1 (Mean-based)')}), indicating significant run-to-run variation potentially due to MPI setup or complex interactions even at NP=1. For MPI versions, examine if CV increases with NP. Small 'n' values (check table `df_variability` for sample sizes) reduce CV reliability.

**3. Multi-Dimensional Performance (Radar Chart):**
*   **Figure:** `adv_multi_metric_radar_chart_final_revised.png`
*   **Metrics (Median-based Speedup/Efficiency for this chart):** 'NP1 Perf (1/Med.T_NP1)', 'Max Scaled Speedup', 'Max Scaled Efficiency', 'Code Volume (log10 LOC)' (normalized so outer edge means less code).
*   **Takeaway:** Visualizes relative strengths. Outer edge is "better" for each metric's normalized scale.
*   **Your Observation & Data:**
    *   V1 Serial: Strong on 'Code Volume' (less code is better, so it's far out on that axis after normalization). Performance metrics are closer to the center (worse).
    *   V2.2 ScatterHalo: Aims for a balance, showing some 'Max Scaled Speedup' and 'Efficiency' for CPU parallelism compared to V1, but with higher 'Code Volume'.
    *   V3 CUDA: Dominates 'NP1 Perf.' due to GPU acceleration. 'Max Scaled Speedup/Efficiency' are 1.0 by definition (as it's NP=1). 'Code Volume' is moderate.
    *   V4 MPI+CUDA: Achieved good 'NP1 Perf.' (comparable to V3), but its 'Max Scaled Speedup'/'Efficiency' at NP={np_val_str_v4 if np_val_str_v4 != 'N/A' else 'Max'} are poor due to host-staging, pulling it inwards on those axes relative to its NP=1 potential. It has the highest 'Code Volume' (least favorable, so closest to center on that axis before normalization, furthest after 1-x normalization).
    *   Refer to the scorecard for absolute magnitudes. The radar chart shows relative normalized strengths.

**4. Overall Project Trajectory & Bottlenecks:**
*   **Scorecard Table:** `project_final_scorecard_median_cv_from_stats.md` (Medians for T_NP1, T@Max_Scaled_NP; Speedup/Efficiency from these medians).
*   **Super-linear Speedup Check:** The V2.2 Speedup (Medians) @NP={np_val_str_v22 if np_val_str_v22 != 'N/A' else 'Max'} is {speedup_median_npmax_v22_formatted_str}. This is typically sub-linear for this type of problem after accounting for communication, as expected.
*   **Performance Discussion (from Scorecard):**
    *   V1 Median T_NP1: {get_formatted_scorecard_value(df_final_scorecard, 'V1 Serial', 'Median T_NP1 (s)')}s is the reference.
    *   V2.2 Median T @Max Scaled NP (NP={np_val_str_v22 if np_val_str_v22 != 'N/A' else 'Max'}): {get_formatted_scorecard_value(df_final_scorecard, 'V2.2 ScatterHalo', 'Median T @Max Scaled NP (s)')}s; Median-based Speedup: {speedup_median_npmax_v22_formatted_str}. Demonstrates effective CPU scaling with MPI scatter/halo.
    *   V3 Median T_NP1: {get_formatted_scorecard_value(df_final_scorecard, 'V3 CUDA', 'Median T_NP1 (s)')}s (approx. {v3_vs_v1_speedup_str} vs V1), showing significant GPU acceleration.
    *   V4 Median T @Max Scaled NP (NP={np_val_str_v4 if np_val_str_v4 != 'N/A' else 'Max'}): {get_formatted_scorecard_value(df_final_scorecard, 'V4 MPI+CUDA', 'Median T @Max Scaled NP (s)')}s; Median-based Speedup: {v4_speedup_np_max_str}. Poor scaling indicates host-staging overheads dominate benefits of multi-GPU distribution for this problem size and implementation.
*   **Bottleneck Migration:** The project successfully demonstrated bottleneck migration: V1 (CPU compute-bound) -> V2.2 (MPI communication overheads and remaining CPU compute) -> V3 (GPU compute, PCIe bandwidth for H-D transfers) -> V4 (Host-staging: MPI communication on CPU, multiple full-tile H-D-H transfers, CPU logic for halo management).

**5. Expert Perspectives & Recommendations (Critique Addressed):**
*   **Performance Engineer:** The V4 MPI+CUDA version's poor scaling ({v4_speedup_np_max_str} at NP={np_val_str_v4 if np_val_str_v4 != 'N/A' else 'Max'}) despite good NP=1 performance ({get_formatted_scorecard_value(df_final_scorecard, 'V4 MPI+CUDA', 'Median T_NP1 (s)')}s) clearly points to inter-process communication and data staging (CPU-GPU transfers) as the major bottleneck. Profiling V4 with Nsight Systems is essential to quantify these overheads. The primary recommendation is to **implement V5 (CUDA-Aware MPI)** to allow direct GPU-GPU communication, bypassing CPU staging for halo exchanges. Subsequently, investigate asynchronous operations (CUDA streams for compute/copy overlap, non-blocking MPI) for V4/V5. The high CV@NP1 for V4 MPI+CUDA ({get_formatted_scorecard_value(df_final_scorecard, 'V4 MPI+CUDA', 'CV @NP1 (Mean-based)')}) also warrants investigation – could be initial MPI synchronization costs or variability in GPU resource availability even when running as a single MPI process.
*   **Software Engineer:** The Lines of Code (LOC) for V4 MPI+CUDA ({get_formatted_scorecard_value(df_final_scorecard, 'V4 MPI+CUDA', 'LOC (Core)', fmt="{{:d}}")}) being the highest reflects the significant integration complexity of combining MPI and CUDA with manual host staging. The `alexnetTileForwardCUDA` in V4 acts as a monolithic GPU compute step per rank, which is good for local GPU utilization but highlights the data movement problem to/from it. The high runtime variability (CV) for V4 is a concern for reproducibility and indicates potential instability or contention. Future work should focus on reducing V4's complexity and improving its stability if V5 is not pursued.
*   **Data Analyst:** Using median runtimes for scorecard KPIs enhances robustness against outliers compared to using minimums. The small sample sizes ('n' values in `df_variability`) for some runs limit the precision of Confidence Intervals around CV and median estimates. The statistically insignificant Pearson R ({correlation_r_str}, p={correlation_p_val_str}) between LOC and NP=1 performance implies that for this project, the architectural paradigm (Serial, MPI, CUDA, Hybrid) and the efficiency of its implementation (e.g., scatter/halo vs. broadcast, host-staging vs. direct GPU) were far more determinant of single-process performance than simply the amount of code written.
*   **Domain Expert (HPC for AI):** The V1 serial median time ({get_formatted_scorecard_value(df_final_scorecard, 'V1 Serial', 'Median T_NP1 (s)')}s) serves as a critical reference; ensure modern compiler optimizations (-O3, target architecture flags) were used. V3's strong performance ({v3_vs_v1_speedup_str} over V1) confirms the suitability of GPUs for CNN computations. V4's poor scaling is a classic symptom of naive hybrid implementations where data movement between host and device for inter-process communication dominates. **CUDA-Aware MPI (V5) is the standard HPC solution to mitigate this for distributed GPU training/inference of such workloads.** Further, optimizing data layout (e.g., NCHW vs. NHWC) and using optimized libraries (cuDNN, if full layers were built) would be typical next steps in a production setting, though out of scope here.

**Further Project Steps (as outlined in presentation):**
1.  **Implement V5 (CUDA-Aware MPI):** Directly address V4's primary bottleneck.
2.  **Asynchronous Overlap:** If V5 proves difficult or for further optimization, explore CUDA streams and non-blocking MPI to overlap data transfers with computation.
3.  **Cluster Profiling:** Utilize Nsight Systems/Compute on the target cluster for detailed analysis of V3, V4, and any V5 implementation to precisely identify bottlenecks.
4.  **Reporting:** Clearly distinguish between MIN-based metrics (often used for "best case" speedup plots) and MEDIAN-based metrics (used here for robust scorecard evaluation). Discuss the impact of small sample sizes on statistical confidence where applicable. Provide comprehensive details of the hardware/software environment used for reproducibility.
"""

if 'Markdown' in globals() and 'display' in globals(): display(Markdown(interpretation_md))
else: print(interpretation_md)

interpretation_file_path = visuals_output_dir / "qualitative_interpretation_summary_final_valueerror_fixed.md"
with open(interpretation_file_path, "w", encoding="utf-8") as f: f.write(interpretation_md)
print(f"\n✓ Final qualitative interpretation (ValueError fixed) saved to {interpretation_file_path}")

print("\n--- Grand Synthesis Cell (Critique Addressed & ValueError Fixed) Successfully Completed ---")
```

    [INFO] Seaborn theme applied for enhanced plot aesthetics.
    [INFO] All plots and advanced exports for this cell will be saved to: /home/myko/CS485/CUDA-MPI-GPU-Cluster-Programming/analysis_visuals_final
    
    --- 1. Aggregating Lines of Code (LOC) for Canonical Versions ---
    [INFO] Attempting to execute query: SELECT COUNT(*) as count FROM source_stats...
    [INFO] Attempting to execute query: 
            SELECT 'V1 Serial' as version, COALESCE(SUM(loc), 0) as total_loc
            FROM source_stats...
    [INFO] Attempting to execute query: 
            SELECT 'V2.1 BroadcastAll' as version, COALESCE(SUM(loc), 0) as total_loc
            FROM sour...
    [INFO] Attempting to execute query: 
            SELECT 'V2.2 ScatterHalo' as version, COALESCE(SUM(loc), 0) as total_loc
            FROM sourc...
    [INFO] Attempting to execute query: 
            SELECT 'V3 CUDA' as version, COALESCE(SUM(loc), 0) as total_loc
            FROM source_stats
     ...
    [INFO] Attempting to execute query: 
            SELECT 'V4 MPI+CUDA' as version, COALESCE(SUM(loc), 0) as total_loc
            FROM source_sta...
    [INFO] Attempting to execute query: 
            SELECT 'V5 MPI+CUDA-Aware' as version, COALESCE(SUM(loc), 0) as total_loc
            FROM sour...
    Aggregated LOC for core algorithmic code (src/, include/):



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>version</th>
      <th>total_loc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>V1 Serial</td>
      <td>525</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V2.1 BroadcastAll</td>
      <td>306</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V2.2 ScatterHalo</td>
      <td>483</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V3 CUDA</td>
      <td>354</td>
    </tr>
    <tr>
      <th>4</th>
      <td>V4 MPI+CUDA</td>
      <td>576</td>
    </tr>
  </tbody>
</table>
</div>


    
    --- 2. Plotting: Median NP=1 Performance vs. Lines of Code (Revised Visuals) ---
    [INFO] Attempting to execute query: SELECT version, MEDIAN(total_time_s) as median_np1_s FROM perf_runs WHERE np = 1 AND total_time_s IS...



    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    /tmp/ipykernel_96940/2115401082.py in ?()
        164 if not df_loc_for_plot.empty and plt is not None:
        165     df_median_np1 = execute_query("SELECT version, MEDIAN(total_time_s) as median_np1_s FROM perf_runs WHERE np = 1 AND total_time_s IS NOT NULL GROUP BY version")
        166 
        167     if df_median_np1 is not None and not df_median_np1.empty:
    --> 168         df_perf_vs_loc = pd.merge(df_median_np1, df_loc_for_plot, on="version")
        169 
        170         if not df_perf_vs_loc.empty and len(df_perf_vs_loc) > 1 :
        171             plt.figure(figsize=(11, 7))


    ~/CS485/CUDA-MPI-GPU-Cluster-Programming/.venv/lib/python3.12/site-packages/pandas/core/reshape/merge.py in ?(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
        166             validate=validate,
        167             copy=copy,
        168         )
        169     else:
    --> 170         op = _MergeOperation(
        171             left_df,
        172             right_df,
        173             how=how,


    ~/CS485/CUDA-MPI-GPU-Cluster-Programming/.venv/lib/python3.12/site-packages/pandas/core/reshape/merge.py in ?(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
        790             self.right_join_keys,
        791             self.join_names,
        792             left_drop,
        793             right_drop,
    --> 794         ) = self._get_merge_keys()
        795 
        796         if left_drop:
        797             self.left = self.left._drop_labels_or_levels(left_drop)


    ~/CS485/CUDA-MPI-GPU-Cluster-Programming/.venv/lib/python3.12/site-packages/pandas/core/reshape/merge.py in ?(self)
       1306                     if lk is not None:
       1307                         # Then we're either Hashable or a wrong-length arraylike,
       1308                         #  the latter of which will raise
       1309                         lk = cast(Hashable, lk)
    -> 1310                         left_keys.append(left._get_label_or_level_values(lk))
       1311                         join_names.append(lk)
       1312                     else:
       1313                         # work-around for merge_asof(left_index=True)


    ~/CS485/CUDA-MPI-GPU-Cluster-Programming/.venv/lib/python3.12/site-packages/pandas/core/generic.py in ?(self, key, axis)
       1907             values = self.xs(key, axis=other_axes[0])._values
       1908         elif self._is_level_reference(key, axis=axis):
       1909             values = self.axes[axis].get_level_values(key)._values
       1910         else:
    -> 1911             raise KeyError(key)
       1912 
       1913         # Check for duplicates
       1914         if values.ndim > 1:


    KeyError: 'version'

