#!/usr/bin/env python3
"""
pair_test_ll24_vs_ll50_full_auto.py

Paired Wilcoxon signed-rank tests for:
    24-ROI LL-only  vs.  50-ROI LL-only
on the full EEG-ImageNet label sets:
    - all
    - coarse
    - fine

This script assumes it is placed inside the same train_paper folder that contains:
    ll_baselines_results/summary_all_subjects_*.csv

It automatically reads the latest summary CSV unless one is provided.

Usage:
    python pair_test_ll24_vs_ll50_full_auto.py

Optional:
    python pair_test_ll24_vs_ll50_full_auto.py --summary-csv ll_baselines_results/summary_all_subjects_0-15.csv
    python pair_test_ll24_vs_ll50_full_auto.py --output-csv paired_ll24_vs_ll50_by_subset.csv
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


SUBSETS = ["all", "coarse", "fine"]


def find_latest_summary(results_dir: Path) -> Path:
    files = sorted(
        results_dir.glob("summary_all_subjects_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No summary_all_subjects_*.csv found in {results_dir}")
    return files[0]


def require_columns(df: pd.DataFrame, cols):
    missing = set(cols) - set(df.columns)
    if missing:
        sys.exit(f"Missing required columns: {sorted(missing)}")


def paired_test_for_subset(df: pd.DataFrame, subset: str):
    df24 = df[(df["subset"] == subset) & (df["feature_set"] == "ll24")][["subject", "accuracy_mean"]].copy()
    df50 = df[(df["subset"] == subset) & (df["feature_set"] == "ll50")][["subject", "accuracy_mean"]].copy()

    df24 = df24.rename(columns={"accuracy_mean": "ll24"})
    df50 = df50.rename(columns={"accuracy_mean": "ll50"})

    merged = pd.merge(df24, df50, on="subject", how="inner").sort_values("subject").reset_index(drop=True)
    if merged.empty:
        raise ValueError(f"No overlapping ll24/ll50 subjects found for subset '{subset}'")

    x = merged["ll24"].to_numpy(dtype=float)
    y = merged["ll50"].to_numpy(dtype=float)
    d = y - x

    stat, p = wilcoxon(y, x, zero_method="wilcox", alternative="two-sided", correction=False, mode="auto")

    summary = {
        "subset": subset,
        "n_subjects": int(len(merged)),
        "ll24_mean": float(np.mean(x)),
        "ll24_sd": float(np.std(x, ddof=1)),
        "ll50_mean": float(np.mean(y)),
        "ll50_sd": float(np.std(y, ddof=1)),
        "mean_paired_diff_ll50_minus_ll24": float(np.mean(d)),
        "median_paired_diff_ll50_minus_ll24": float(np.median(d)),
        "wilcoxon_statistic": float(stat),
        "wilcoxon_pvalue": float(p),
    }
    return summary, merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=str, default=None,
                        help="Optional path to ll_baselines_results/summary_all_subjects_*.csv")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Optional path to save merged subject-level rows across subsets")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    results_dir = here / "ll_baselines_results"
    summary_csv = Path(args.summary_csv) if args.summary_csv else find_latest_summary(results_dir)

    if not summary_csv.exists():
        sys.exit(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)
    require_columns(df, {"subject", "subset", "feature_set", "accuracy_mean"})

    all_summaries = []
    merged_tables = []

    print("\n=== Paired comparison: 24-ROI LL-only vs 50-ROI LL-only ===")
    print(f"Input summary: {summary_csv}")

    for subset in SUBSETS:
        summary, merged = paired_test_for_subset(df, subset)
        all_summaries.append(summary)

        merged = merged.copy()
        merged.insert(1, "subset", subset)
        merged["diff_ll50_minus_ll24"] = merged["ll50"] - merged["ll24"]
        merged_tables.append(merged)

        print(f"\n--- Subset: {subset} ---")
        print(f"Subjects included: {summary['n_subjects']}")
        print(f"24-ROI mean ± SD:      {summary['ll24_mean']:.4f} ± {summary['ll24_sd']:.4f}")
        print(f"50-ROI mean ± SD:      {summary['ll50_mean']:.4f} ± {summary['ll50_sd']:.4f}")
        print(f"Mean paired diff:      {summary['mean_paired_diff_ll50_minus_ll24']:.4f}   (ll50 - ll24)")
        print(f"Median paired diff:    {summary['median_paired_diff_ll50_minus_ll24']:.4f} (ll50 - ll24)")
        print(f"Wilcoxon statistic:    {summary['wilcoxon_statistic']:.4f}")
        print(f"Wilcoxon p-value:      {summary['wilcoxon_pvalue']:.6f}")
        if summary["wilcoxon_pvalue"] < 0.05:
            print("Interpretation: a statistically supported difference was detected.")
        else:
            print("Interpretation: no statistically supported improvement was observed.")

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(merged_tables, ignore_index=True).to_csv(out_path, index=False)
        print(f"\nSaved merged subject-level table to: {out_path}")

    # also save a summary CSV next to requested output if given
    if args.output_csv:
        summary_out = Path(args.output_csv).with_name(Path(args.output_csv).stem + "_summary.csv")
        pd.DataFrame(all_summaries).to_csv(summary_out, index=False)
        print(f"Saved subset summary table to: {summary_out}")


if __name__ == "__main__":
    main()
