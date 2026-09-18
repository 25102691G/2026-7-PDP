#!/usr/bin/env python3
"""Validate a manifest-scoped summary against PrivIM Table II checkpoints."""

import argparse
from pathlib import Path

import pandas as pd


DATASETS = ("email", "bitcoin", "lastfm", "hepph", "facebook", "gowalla")
PAPER = {
    (1, "PrivIM"): (62.01, 60.13, 31.72, 32.24, 70.20, 68.05),
    (1, "PrivIM+SCS"): (83.05, 85.76, 80.72, 80.98, 82.68, 81.28),
    (1, "PrivIM+"): (83.87, 86.42, 83.50, 85.49, 85.10, 86.57),
    (4, "PrivIM"): (76.08, 73.48, 79.39, 73.54, 82.60, 82.26),
    (4, "PrivIM+SCS"): (95.08, 92.24, 89.56, 93.91, 90.92, 95.47),
    (4, "PrivIM+"): (94.44, 92.84, 93.76, 94.50, 95.55, 98.02),
    (10000, "PrivIM+"): (98.09, 98.99, 97.93, 98.42, 97.50, 99.78),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--expected-runs", type=int, default=5)
    parser.add_argument("--max-std", type=float, default=5.0)
    parser.add_argument("--max-non-private-mae", type=float, default=3.0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.summary)
    frame["dataset"] = frame["graph"].str.replace(
        "_test_edges.txt", "", regex=False
    )
    frame["epsnum"] = frame["epsilon"].str.replace(
        "eps_", "", regex=False
    ).astype(int)

    problems = []
    if frame.isna().any().any():
        problems.append("summary contains missing values")
    incomplete = frame[frame["runs"] != args.expected_runs]
    if not incomplete.empty:
        problems.append(f"{len(incomplete)} configurations have incomplete runs")
    unstable = frame[
        frame["coverage_ratio_std"] > args.max_std
    ][["dataset", "setting", "epsilon", "coverage_ratio_std"]]
    if not unstable.empty:
        problems.append(
            f"{len(unstable)} configurations exceed std={args.max_std:g}"
        )

    comparisons = []
    for (epsilon, setting), expected in PAPER.items():
        for dataset, paper_value in zip(DATASETS, expected):
            row = frame[
                (frame["dataset"] == dataset)
                & (frame["setting"] == setting)
                & (frame["epsnum"] == epsilon)
            ]
            if row.empty:
                continue
            current = float(row.iloc[0]["coverage_ratio_mean"])
            comparisons.append({
                "epsilon": epsilon,
                "setting": setting,
                "dataset": dataset,
                "current": current,
                "paper": paper_value,
                "absolute_error": abs(current - paper_value),
            })
    comparison = pd.DataFrame(comparisons)
    if not comparison.empty:
        print(
            comparison.groupby(["epsilon", "setting"])
            .agg(
                current_mean=("current", "mean"),
                paper_mean=("paper", "mean"),
                mae=("absolute_error", "mean"),
            )
            .round(2)
            .to_string()
        )
        non_private = comparison[comparison["epsilon"] == 10000]
        if (
            not non_private.empty
            and non_private["absolute_error"].mean() > args.max_non_private_mae
        ):
            problems.append(
                "non-private mean absolute error exceeds "
                f"{args.max_non_private_mae:g} points"
            )

    if not unstable.empty:
        print("\nUnstable configurations:")
        print(unstable.sort_values("coverage_ratio_std", ascending=False).to_string(index=False))
    if problems:
        print("\nValidation warnings:")
        for problem in problems:
            print(f"- {problem}")
    else:
        print("\nValidation passed.")
    if args.strict and problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
