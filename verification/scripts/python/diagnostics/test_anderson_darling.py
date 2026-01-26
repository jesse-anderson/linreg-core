#!/usr/bin/env python3
# ============================================================================
# Anderson-Darling Test Reference Implementation (Python)
# ============================================================================
# This script generates reference values for the Anderson-Darling test using
# Python's statsmodels library. The Anderson-Darling test checks whether
# residuals are normally distributed, with particular sensitivity to tail
# deviations.
#
# Source: statsmodels.stats.diagnostic.normal_ad
# Reference: Anderson & Darling (1952), "Asymptotic theory of certain
#            goodness of fit criteria based on stochastic processes"
#            Stephens (1974), "EDF Statistics for Goodness of Fit"
#
# Usage:
#   python test_anderson_darling.py --csv <csv_path> [--output <output_dir>]
#
# Args:
#   --csv      Path to CSV file (first col = response, rest = predictors)
#              Default: ../../datasets/csv/mtcars.csv
#   --output   Path to output directory
#              Default: ../../results/python
# ============================================================================

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import normal_ad


def convert_categorical_to_numeric(df, dataset_name):
    """Convert categorical columns to numeric representations."""
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if non_numeric_cols:
        print(f"INFO: Dataset '{dataset_name}' contains non-numeric columns: "
              f"{', '.join(non_numeric_cols)}")
        print("Converting categorical variables to numeric representations...")

        for col in non_numeric_cols:
            # Use factorize for reliable categorical encoding
            df[col], uniques = pd.factorize(df[col])
            print(f"  {col}: {len(uniques)} unique values -> integer level encoding")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Run Anderson-Darling test using statsmodels'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='../../datasets/csv/mtcars.csv',
        help='Path to CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../../results/python',
        help='Path to output directory (deprecated, use --output-dir)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../../results/python',
        help='Path to output directory'
    )

    args = parser.parse_args()
    csv_path = Path(args.csv)
    # Use --output-dir if provided, otherwise fall back to --output
    output_path = args.output_dir if hasattr(args, 'output_dir') and args.output_dir != '../../results/python' else args.output
    output_dir = Path(output_path)

    # Validate CSV path
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    # Extract dataset name from filename
    dataset_name = csv_path.stem

    # Read CSV data
    data = pd.read_csv(csv_path)

    # Convert categorical columns to numeric
    data = convert_categorical_to_numeric(data, dataset_name)

    # Assume first column is response variable, rest are predictors
    response_col = data.columns[0]
    predictor_cols = data.columns[1:].tolist()

    # Build formula string
    formula_str = f"{response_col} ~ {' + '.join(predictor_cols)}"

    # Fit the model using statsmodels
    import statsmodels.api as sm
    X = data[predictor_cols]
    X = sm.add_constant(X)  # Add intercept
    y = data[response_col]

    model = sm.OLS(y, X).fit()
    residuals = model.resid

    # Run Anderson-Darling test on residuals
    # Note: statsmodels returns (statistic, p_value)
    ad_stat, p_value = normal_ad(residuals)

    # Handle infinite/NaN values from statsmodels (can occur with high-dimensional data)
    import numpy as np
    if np.isinf(ad_stat) or np.isnan(ad_stat):
        ad_stat = 999.999  # Large placeholder value indicating extreme non-normality
    if np.isinf(p_value) or np.isnan(p_value):
        p_value = 0.0  # p-value of 0 means reject H0 (non-normal)

    # Print results
    print("Anderson-Darling Test (Python - statsmodels.stats.diagnostic.normal_ad)")
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Formula: {formula_str}")
    print(f"A-statistic: {ad_stat:.22g}")
    print(f"p-value: {p_value:.22g}")
    print(f"Passed: {p_value > 0.05}")
    print()

    # Prepare output
    output = {
        "test_name": "Anderson-Darling Test (Python - statsmodels.stats.diagnostic.normal_ad)",
        "dataset": dataset_name,
        "formula": formula_str,
        "statistic": ad_stat,
        "p_value": p_value,
        "passed": bool(p_value > 0.05),
        "description": "Tests for normality of residuals. The Anderson-Darling test is "
                      "particularly sensitive to deviations in the tails of the distribution. "
                      "Uses statsmodels.stats.diagnostic.normal_ad."
    }

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON with naming convention: {dataset}_anderson_darling.json
    output_file = output_dir / f"{dataset_name}_anderson_darling.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, allow_nan=False)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
