#!/usr/bin/env python3
# ============================================================================
# Shapiro-Wilk Test Validation Script (Python)
# ============================================================================
#
# This script runs the Shapiro-Wilk test on regression residuals using
# scipy's shapiro function and outputs the results to JSON for validation
# against the Rust implementation.
#
# Usage: python test_shapiro_wilk.py --csv <csv_file>
# Example: python test_shapiro_wilk.py --csv ../../datasets/csv/mtcars.csv
#
# Output: JSON file with test results

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

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

def parse_args():
    parser = argparse.ArgumentParser(description='Shapiro-Wilk test validation')
    parser.add_argument('--csv', required=True, help='Path to CSV file')
    parser.add_argument('--output-dir', default='../../results/python', help='Path to output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    csv_file = args.csv

    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}", file=sys.stderr)
        sys.exit(1)

    # Get dataset name
    dataset_name = os.path.splitext(os.path.basename(csv_file))[0]

    # Read CSV data using pandas to handle categorical columns
    data = pd.read_csv(csv_file)

    # Convert categorical columns to numeric
    data = convert_categorical_to_numeric(data, dataset_name)

    # Convert to numpy array
    data = data.values

    # First column is y (dependent variable), rest are x variables (predictors)
    y = data[:, 0]
    if data.shape[1] > 1:
        x_vars = data[:, 1:]
    else:
        # No predictors - simple mean model
        x_vars = None

    # Fit OLS model and compute residuals
    if x_vars is None:
        # Simple mean model
        residuals = y - np.mean(y)
    else:
        # OLS regression
        # Add intercept column
        X = np.column_stack([np.ones(len(y)), x_vars])

        # Solve normal equations: beta = (X'X)^(-1) X'y
        try:
            beta = np.linalg.solve(X.T @ X, X.T @ y)
        except np.linalg.LinAlgError:
            # Use least squares if singular
            beta = np.linalg.lstsq(X, y, rcond=None)[0]

        # Compute residuals
        residuals = y - X @ beta

    # Run Shapiro-Wilk test using scipy's native function
    # H0: Residuals are normally distributed
    # H1: Residuals are not normally distributed
    w_statistic, p_value = stats.shapiro(residuals)

    # Determine if test passed (null hypothesis not rejected at alpha = 0.05)
    alpha = 0.05
    passed = p_value > alpha

    # Create interpretation text
    if passed:
        interpretation = (
            f"p-value = {p_value:.4f} is greater than {alpha:.2f}. "
            f"Cannot reject H0. No significant evidence that residuals deviate from normality."
        )
        guidance = ("The normality assumption appears to be met. Shapiro-Wilk test does not detect "
                    "significant deviation from normal distribution.")
    else:
        interpretation = (
            f"p-value = {p_value:.4f} is less than or equal to {alpha:.2f}. "
            f"Reject H0. Significant evidence that residuals deviate from normality."
        )
        guidance = ("Consider transforming the dependent variable (e.g., log, Box-Cox transformation), "
                    "using robust standard errors, or applying a different estimation method.")

    # Create output dictionary
    output = {
        "test_name": "Shapiro-Wilk Test for Normality",
        "statistic": float(w_statistic),
        "p_value": float(p_value),
        "is_passed": bool(passed),
        "interpretation": interpretation,
        "guidance": guidance
    }

    # Get output file name from CSV file name
    basename = os.path.splitext(os.path.basename(csv_file))[0]
    output_file = os.path.join(args.output_dir, f"{basename}_shapiro_wilk.json")

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Write output to JSON file
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results written to: {output_file}")
    print(f"W statistic: {w_statistic}")
    print(f"p-value: {p_value}")

if __name__ == '__main__':
    main()
