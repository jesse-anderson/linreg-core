"""
LOESS Test Reference Implementation (Python with statsmodels)

This script generates reference values for LOESS regression using statsmodels'
lowess function. The test validates that the Rust implementation matches
Python's behavior for fitted values.

Source: statsmodels.nonparametric.smoothers_lowess.lowess
Reference: Cleveland, W. S. (1979). "Robust Locally Weighted Regression
          and Smoothing Scatterplots". JASA, 74(368), 829-836.

Usage:
    python test_loess.py [csv_path] [output_dir]
    Args:
        csv_path   - Path to CSV file (first col = response, rest = predictors)
                     Default: ../../../datasets/csv/faithful.csv
        output_dir - Path to output directory
                     Default: ../../../results/python
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def resolve_path(path: str, base_dir: str) -> str:
    """Resolve path relative to base directory."""
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def convert_to_numeric(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Convert categorical columns to numeric."""
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if non_numeric_cols:
        print(f"INFO: Dataset '{dataset_name}' contains non-numeric columns: "
              f"{', '.join(non_numeric_cols)}")
        print("Converting categorical variables to numeric representations...")

        for col in non_numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # If still has NaNs, use factor codes
            if df[col].isna().any():
                df[col] = df[col].astype('category').cat.codes

    return df


def loess_lowess(x, y, span):
    """
    LOESS/LOWESS with degree 1 (linear) using statsmodels.

    Note: statsmodels.lowess is LOWESS (Locally Weighted Scatterplot Smoothing)
    which uses degree 1 by design. This is not a full LOESS implementation -
    it cannot fit quadratic or higher degree polynomials.

    For comparison:
    - R's loess() supports degrees 0, 1, and 2
    - statsmodels.lowess only supports degree 1
    - statsmodels.lowess always uses direct fitting (no interpolation)
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess
    return lowess(y, x, frac=span, return_sorted=False)


def main():
    parser = argparse.ArgumentParser(description='Generate LOESS reference values')
    parser.add_argument('csv_path', nargs='?',
                        default='../../../datasets/csv/faithful.csv',
                        help='Path to CSV file')
    parser.add_argument('output_dir', nargs='?',
                        default='../../../results/python',
                        help='Output directory')

    args = parser.parse_args()

    # Use paths as provided by the runner (absolute paths from repo root)
    csv_path = args.csv_path
    output_dir = args.output_dir

    # Validate parameters
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    # Extract dataset name
    dataset_name = Path(csv_path).stem

    print(f"Running LOESS regression on dataset: {dataset_name}")

    # Load data
    df = pd.read_csv(csv_path)
    df = convert_to_numeric(df, dataset_name)

    # Extract response and predictors
    y = df.iloc[:, 0].values
    x_all = df.iloc[:, 1:].values

    # Use first predictor only (LOESS is primarily for single predictor)
    n_predictors = x_all.shape[1]
    if n_predictors > 1:
        print(f"Note: Using first predictor only (LOESS single-predictor focus)")
    x = x_all[:, 0]

    n = len(y)

    # Test configurations: 3 spans for degree 1 only (LOWESS limitation)
    # Note: statsmodels.lowess is LOWESS (degree 1) by design, not full LOESS
    test_configs = [
        (0.25, 1),
        (0.50, 1),
        (0.75, 1),
    ]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    for span, degree in test_configs:
        print(f"  Testing span={span:.2f} degree={degree} (LOWESS)...")

        # Use statsmodels for LOWESS (degree 1 only)
        fitted = loess_lowess(x, y, span)
        method = "statsmodels_lowess"

        # Prepare output
        result = {
            'test': 'loess',
            'method': method,
            'dataset': dataset_name,
            'n': int(n),
            'n_predictors': 1,
            'span': span,
            'degree': degree,
            'surface': 'direct',  # statsmodels lowess always uses direct fitting
            'fitted': fitted.tolist(),
            'y': y.tolist(),
            'x': x.tolist()
        }

        # Write output
        output_file = os.path.join(output_dir, f"{dataset_name}_loess_{span:.2f}_d{degree}.json")

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"    Wrote: {os.path.basename(output_file)}")

    print(f"Done: {dataset_name} (3 outputs - LOWESS degree 1 only)")


if __name__ == '__main__':
    main()
