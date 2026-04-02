#!/usr/bin/env python3
"""
Aggregate CSV files for a given experiment name.

Example directory layout:
    /global/home/users/jenniferzhao/scratch/aorl2/<experiment_name>/<run_dir>/*.csv

Example:
    /global/home/users/jenniferzhao/scratch/aorl2/2026-04-01-00/
        2026-04-01-00.01e825d00e5393d9ac9643f2a77c2e41faa1116219802f7a414f746b5ce40cb5/
            metrics.csv
        2026-04-01-00.abc123.../
            metrics.csv

Usage examples:

1) Run directly on the remote machine:
    python aggregate_csvs.py 2026-04-01-00 --csv-pattern '*.csv' --group-by step

2) Run from your laptop through ssh:
    python aggregate_csvs.py 2026-04-01-00 --host my.cluster.edu --csv-pattern '*.csv' --group-by step

3) Restrict to one filename:
    python aggregate_csvs.py 2026-04-01-00 --csv-pattern 'progress.csv' --group-by step

Outputs:
    <experiment_name>_all.csv
    <experiment_name>_agg.csv
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


# DEFAULT_BASE_DIR = "/global/home/users/jenniferzhao/scratch/aorl2"
DEFAULT_BASE_DIR = "~/scratch/aorl2"


def run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the completed process."""
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def run_shell_capture(command: str) -> str:
    """Run a shell command locally and capture stdout."""
    proc = subprocess.run(
        command,
        shell=True,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        executable="/bin/bash",
    )
    return proc.stdout


def ssh_capture(host: str, remote_command: str) -> str:
    """Run a command on a remote host over ssh and capture stdout."""
    proc = run_cmd(["ssh", host, remote_command])
    return proc.stdout


def list_csv_files(
    experiment_name: str,
    base_dir: str,
    csv_pattern: str,
    host: Optional[str] = None,
) -> List[str]:
    """
    Find CSV files under:
        <base_dir>/<experiment_name>/*/<csv_pattern>
    """
    exp_dir = f"{base_dir.rstrip('/')}/{experiment_name}"

    # We only search one level below the experiment directory:
    #   exp_dir/<run_dir>/<csv files>
    # If your CSVs are deeper, remove -maxdepth 2 or increase it.
    remote_find = (
        f"find {shlex.quote(exp_dir)} -maxdepth 2 -type f -name {shlex.quote(csv_pattern)} | sort"
    )

    try:
        output = ssh_capture(host, remote_find) if host else run_shell_capture(remote_find)
    except subprocess.CalledProcessError as e:
        print("Failed to list CSV files.", file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        raise

    files = [line.strip() for line in output.splitlines() if line.strip()]
    return files


def extract_run_dir_name(path: str, experiment_name: str) -> str:
    """
    Extract run dir name from:
        .../<experiment_name>/<run_dir>/<file.csv>
    """
    parts = Path(path).parts
    try:
        exp_idx = parts.index(experiment_name)
        return parts[exp_idx + 1]
    except (ValueError, IndexError):
        return "unknown_run"


def read_csv_local(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def read_csv_remote(host: str, path: str) -> pd.DataFrame:
    """
    Read a remote CSV by streaming it through ssh.
    """
    remote_cmd = f"cat {shlex.quote(path)}"
    proc = run_cmd(["ssh", host, remote_cmd])
    return pd.read_csv(io.StringIO(proc.stdout))


def read_all_csvs(
    csv_paths: Iterable[str],
    experiment_name: str,
    host: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read all CSVs and concatenate them into one dataframe.
    Adds:
        - source_csv
        - run_dir
        - experiment_name
    """
    dfs = []
    for i, path in enumerate(csv_paths, start=1):
        print(f"[{i}] Reading {path}", file=sys.stderr)
        try:
            df = read_csv_remote(host, path) if host else read_csv_local(path)
        except Exception as e:
            print(f"Skipping {path}: {e}", file=sys.stderr)
            continue

        df = df.copy()
        df["source_csv"] = path
        df["run_dir"] = extract_run_dir_name(path, experiment_name)
        df["experiment_name"] = experiment_name
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No CSV files could be read.")

    return pd.concat(dfs, ignore_index=True, sort=False)


def infer_group_by_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try common training-progress columns.
    """
    candidates = ["step", "global_step", "iteration", "iter", "epoch", "env_step"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def aggregate_dataframe(
    df: pd.DataFrame,
    group_by: Optional[str],
) -> pd.DataFrame:
    """
    Aggregate numeric columns by the chosen key.
    Produces mean/std/count for numeric columns.
    """
    if group_by is None:
        group_by = infer_group_by_column(df)

    if group_by is None:
        raise ValueError(
            "Could not infer a group-by column. Please pass --group-by explicitly."
        )

    if group_by not in df.columns:
        raise ValueError(f"Group-by column '{group_by}' not found in CSV columns.")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != group_by]

    if not numeric_cols:
        raise ValueError("No numeric columns found to aggregate.")

    agg = df.groupby(group_by)[numeric_cols].agg(["mean", "std", "count"])
    agg.columns = ["{}_{}".format(col, stat) for col, stat in agg.columns]
    agg = agg.reset_index()
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", help="Example: 2026-04-01-00")
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help=f"Base experiment directory (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--csv-pattern",
        default="*.csv",
        help="Filename pattern for CSVs, e.g. '*.csv' or 'progress.csv'",
    )
    parser.add_argument(
        "--group-by",
        default=None,
        help="Column to aggregate by, e.g. step, epoch, iteration",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Remote SSH host. If omitted, files are read locally.",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Directory to write outputs",
    )

    args = parser.parse_args()

    csv_files = list_csv_files(
        experiment_name=args.experiment_name,
        base_dir=args.base_dir,
        csv_pattern=args.csv_pattern,
        host=args.host,
    )

    if not csv_files:
        print("No matching CSV files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files.", file=sys.stderr)

    all_df = read_all_csvs(
        csv_paths=csv_files,
        experiment_name=args.experiment_name,
        host=args.host,
    )

    agg_df = aggregate_dataframe(all_df, group_by=args.group_by)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_path = outdir / f"{args.experiment_name}_all.csv"
    agg_path = outdir / f"{args.experiment_name}_agg.csv"

    all_df.to_csv(all_path, index=False)
    agg_df.to_csv(agg_path, index=False)

    print(f"Wrote concatenated CSV to: {all_path}")
    print(f"Wrote aggregated CSV to:  {agg_path}")


if __name__ == "__main__":
    main()