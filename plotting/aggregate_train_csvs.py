#!/usr/bin/env python3
"""
Aggregate PATH/*/train.csv files and plot selected metrics.

Example:
    python plotting/aggregate_train_csvs.py --path ~/scratch/aorl2/2026-04-01-00

Outputs:
    <label>_train_all.csv
    <label>_train_agg.csv
    <label>_metrics.png
    <label>_<metric_name>.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Aggregate PATH/*/train.csv files and plot selected metrics.'
    )
    parser.add_argument(
        '--path',
        required=True,
        help='Base path to search under. The script matches PATH/*/train.csv.',
    )
    parser.add_argument(
        '--group-by',
        default=None,
        help='Column to aggregate by. Defaults to an inferred progress column such as step.',
    )
    parser.add_argument(
        '--pattern',
        default='train.csv',
        help='CSV filename to match inside each run directory (default: train.csv).',
    )
    parser.add_argument(
        '--outdir',
        default='output/',
        help='Directory to write outputs into.',
    )
    parser.add_argument(
        '--label',
        default=None,
        help='Prefix for output filenames. Defaults to the final component of --path.',
    )
    parser.add_argument(
        '--metric',
        default='actor/q_loss',
        help=(
            'Metric name to plot. Matches an exact column first, then columns containing '
            'the provided text. Default: actor/q_loss'
        ),
    )
    return parser.parse_args()


def infer_group_by_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ['step', 'global_step', 'iteration', 'iter', 'epoch', 'env_step']
    for column in candidates:
        if column in df.columns:
            return column
    return None


def find_train_csvs(base_path: Path, pattern: str) -> List[Path]:
    return sorted(path for path in base_path.glob(f'*/{pattern}') if path.is_file())


def metric_sort_key(column_name: str) -> tuple[int, str]:
    match = re.search(r'(\d+)$', column_name)
    suffix = int(match.group(1)) if match else -1
    return (suffix, column_name)


def find_metric_columns(df: pd.DataFrame, metric_name: str) -> List[str]:
    exact_matches = [column for column in df.columns if column == metric_name]
    if exact_matches:
        return exact_matches

    substring_matches = [column for column in df.columns if metric_name in column]
    return sorted(substring_matches, key=metric_sort_key)


def sanitize_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('_') or 'metric'


def read_all_csvs(csv_paths: List[Path], label: str) -> pd.DataFrame:
    dataframes = []
    for index, csv_path in enumerate(csv_paths, start=1):
        print(f'[{index}] Reading {csv_path}', file=sys.stderr)
        dataframe = pd.read_csv(csv_path)
        dataframe = dataframe.copy()
        dataframe['source_csv'] = str(csv_path)
        dataframe['run_dir'] = csv_path.parent.name
        dataframe['experiment_name'] = label
        dataframes.append(dataframe)

    if not dataframes:
        raise RuntimeError('No CSV files could be read.')

    return pd.concat(dataframes, ignore_index=True, sort=False)


def aggregate_dataframe(
    df: pd.DataFrame,
    group_by: str,
    metric_columns: List[str],
) -> pd.DataFrame:
    aggregated = df.groupby(group_by)[metric_columns].agg(['mean', 'std', 'count'])
    aggregated.columns = [f'{column}_{stat}' for column, stat in aggregated.columns]
    return aggregated.reset_index().sort_values(group_by)


def plot_metric(
    agg_df: pd.DataFrame,
    group_by: str,
    metric_column: str,
    output_path: Path,
    title: str,
) -> None:
    mean_column = f'{metric_column}_mean'
    std_column = f'{metric_column}_std'

    x_values = agg_df[group_by]
    mean_values = agg_df[mean_column]
    std_values = agg_df[std_column].fillna(0)

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, mean_values, label='mean', linewidth=2)
    plt.fill_between(
        x_values,
        mean_values - std_values,
        mean_values + std_values,
        alpha=0.2,
        label='std',
    )
    plt.xlabel(group_by)
    plt.ylabel(metric_column)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_combined_metrics(
    agg_df: pd.DataFrame,
    group_by: str,
    metric_columns: List[str],
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(10, 6))
    for metric_column in metric_columns:
        plt.plot(
            agg_df[group_by],
            agg_df[f'{metric_column}_mean'],
            linewidth=2,
            label=metric_column,
        )
    plt.xlabel(group_by)
    plt.ylabel('metric value')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()

    base_path = Path(args.path).expanduser().resolve()
    label = args.label or base_path.name
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    csv_paths = find_train_csvs(base_path, args.pattern)
    if not csv_paths:
        print(
            f'No matching CSV files found under {base_path} with pattern */{args.pattern}.',
            file=sys.stderr,
        )
        sys.exit(1)

    print(f'Found {len(csv_paths)} CSV files.', file=sys.stderr)
    all_df = read_all_csvs(csv_paths, label=label)

    group_by = args.group_by or infer_group_by_column(all_df)
    if group_by is None:
        raise ValueError(
            'Could not infer a group-by column. Please pass --group-by explicitly.'
        )
    if group_by not in all_df.columns:
        raise ValueError(f"Group-by column '{group_by}' not found in the aggregated CSV data.")

    metric_columns = find_metric_columns(all_df, args.metric)
    if not metric_columns:
        raise ValueError(
            f"No columns matching '{args.metric}' were found in the aggregated CSV data."
        )

    numeric_metric_columns = [
        column for column in metric_columns if pd.api.types.is_numeric_dtype(all_df[column])
    ]
    if not numeric_metric_columns:
        raise ValueError(
            f"Columns matching '{args.metric}' were found, but none are numeric."
        )

    agg_df = aggregate_dataframe(
        all_df,
        group_by=group_by,
        metric_columns=numeric_metric_columns,
    )

    all_csv_path = outdir / f'{label}_train_all.csv'
    agg_csv_path = outdir / f'{label}_train_agg.csv'
    combined_plot_path = outdir / f'{label}_metrics.png'

    all_df.to_csv(all_csv_path, index=False)
    agg_df.to_csv(agg_csv_path, index=False)

    plot_combined_metrics(
        agg_df,
        group_by=group_by,
        metric_columns=numeric_metric_columns,
        output_path=combined_plot_path,
        title=f'{label}: {args.metric} metrics',
    )

    for metric_column in numeric_metric_columns:
        metric_plot_path = outdir / f'{label}_{sanitize_filename(metric_column)}.png'
        plot_metric(
            agg_df,
            group_by=group_by,
            metric_column=metric_column,
            output_path=metric_plot_path,
            title=f'{label}: {metric_column}',
        )

    print(f'Wrote concatenated CSV to: {all_csv_path}')
    print(f'Wrote aggregated CSV to:  {agg_csv_path}')
    print(f'Wrote combined plot to:   {combined_plot_path}')


if __name__ == '__main__':
    main()
