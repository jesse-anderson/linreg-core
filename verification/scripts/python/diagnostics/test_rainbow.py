# ============================================================================
# Rainbow Test Reference Implementation (Python)
# ============================================================================
# This script generates reference values for the Rainbow test using
# statsmodels.stats.diagnostic.linear_rainbow. The Rainbow test checks for
# linearity by comparing the fit on a central subset of observations against
# the fit on all observations.
#
# Source: statsmodels package, linear_rainbow function
# Reference: Utts (1982), "The Rainbow Test for Lack of Fit in Regression"
#
# Usage:
#   python test_rainbow.py [--csv CSV_PATH] [--output-dir OUTPUT_DIR] [--fraction FRACTION]
#
#   Args:
#     --csv       Path to CSV file (first col = response, rest = predictors)
#                 Default: ../../datasets/csv/mtcars.csv
#     --output-dir Path to output directory
#                 Default: ../../results/python
#     --fraction  Fraction of data for central subset
#                 Default: 0.5
# ============================================================================

import argparse
import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_rainbow


def validate_for_regression(data, dataset_name):
    """Validate data for regression analysis."""
    issues = []

    # Check for non-numeric columns
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        issues.append(f"Non-numeric columns detected: {non_numeric_cols}")

    # Check for missing values
    missing_counts = data.isnull().sum()
    if missing_counts.sum() > 0:
        missing_cols = missing_counts[missing_counts > 0].index.tolist()
        issues.append(f"Missing values detected in: {missing_cols}")

    return issues


def convert_categorical_to_numeric(data, dataset_name):
    """Convert categorical variables to numeric representations."""
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

    if not non_numeric_cols:
        return data, []

    print(f"INFO: Dataset '{dataset_name}' contains non-numeric columns: {non_numeric_cols}")
    print(f"Converting categorical variables to numeric representations...")

    # Process each non-numeric column
    for col in non_numeric_cols:
        if data[col].dtype == 'object':
            # For string/categorical data, use integer encoding
            data[col] = pd.to_numeric(data[col], errors='coerce')
            # Replace NaN (missing values from coercing) with mode
            if data[col].isnull().any():
                mode_val = data[col].mode()[0]
                data[col].fillna(mode_val, inplace=True)
                print(f"  {col}: {len(data[col].unique())} unique values -> integer encoding (missing filled with mode: {mode_val})")
            else:
                print(f"  {col}: {len(data[col].unique())} unique values -> integer encoding")
        else:
            # Already numeric or other type
            pass

    return non_numeric_cols


def main():
    parser = argparse.ArgumentParser(
        description="Rainbow Test - Check linearity assumption using statsmodels"
    )
    parser.add_argument(
        "--csv",
        default="../../datasets/csv/mtcars.csv",
        help="Path to CSV file (first column = response, rest = predictors)"
    )
    parser.add_argument(
        "--output-dir",
        default="../../results/python",
        help="Path to output directory"
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.5,
        help="Fraction of data for central subset (default: 0.5)"
    )

    args = parser.parse_args()

    # Validate CSV path
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    # Extract dataset name from filename
    dataset_name = os.path.splitext(os.path.basename(args.csv))[0]

    # Read CSV data
    data = pd.read_csv(args.csv)

    # Validate data for regression
    issues = validate_for_regression(data, dataset_name)

    # If there are non-numeric columns, convert them to numeric
    if any("Non-numeric" in issue for issue in issues):
        non_numeric_cols = convert_categorical_to_numeric(data, dataset_name)

    # After conversion, re-check if we still have non-numeric columns
    remaining_non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
    if remaining_non_numeric:
        # These could be things like empty strings or other non-numeric types that couldn't be converted
        raise ValueError(f"Could not convert the following non-numeric columns to numeric: {remaining_non_numeric}")

    # Now handle the response variable
    response_col = data.columns[0]

    if pd.api.types.is_numeric_dtype(data[response_col]):
        # Response is numeric, use all other numeric columns as predictors
        predictor_cols = data.columns[1:].tolist()
        print(f"Response variable: {response_col} (numeric)")

    else:
        # Response is categorical - use one-hot encoding
        print(f"INFO: Response variable '{response_col}' is categorical - using one-hot encoding")
        data = pd.get_dummies(data, columns=[response_col], drop_first=True)

        # Update response_col to the first one-hot encoded column (e.g., "Species_I.versicolor")
        response_col = data.columns[0]
        predictor_cols = data.columns[1:].tolist()

        print(f"Predicting '{response_col}' vs all other categories (one-hot encoding)")

    # Prepare data for statsmodels (add constant for intercept)
    X = data[predictor_cols]
    X = sm.add_constant(X)
    y = data[response_col]

    # Build formula string for output
    formula_str = f"{response_col} ~ {' + '.join(predictor_cols)}"

    # Fit the model
    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        raise ValueError(f"Failed to fit regression model: {e}")

    # Run Rainbow test
    try:
        rainbow_result = linear_rainbow(model, frac=args.fraction)
    except Exception as e:
        # Check if it's a multicollinearity issue
        if "singular" in str(e).lower() or "multicollinearity" in str(e).lower():
            output = {
                "test_name": "Rainbow Test (Python - statsmodels)",
                "dataset": dataset_name,
                "formula": formula_str,
                "statistic": None,
                "p_value": None,
                "passed": None,
                "skipped": True,
                "reason": "High multicollinearity detected - cannot reliably perform Rainbow test",
                "description": "Tests for linearity by comparing fit on central subset vs full data. Uses statsmodels.stats.diagnostic.linear_rainbow."
            }
        else:
            raise
    else:
        # Print results
        print("Rainbow Test (Python - statsmodels)")
        print("=" * 40)
        print(f"Dataset: {dataset_name}")
        print(f"Formula: {formula_str}")
        print(f"Fraction: {args.fraction}")
        print(f"Statistic (F): {rainbow_result[0]}")
        print(f"p-value: {rainbow_result[1]}")
        print(f"Passed: {rainbow_result[1] > 0.05}")
        print()

        # Prepare output
        output = {
            "test_name": "Rainbow Test (Python - statsmodels)",
            "dataset": dataset_name,
            "formula": formula_str,
            "statistic": float(rainbow_result[0]),
            "p_value": float(rainbow_result[1]),
            "passed": bool(rainbow_result[1] > 0.05),
            "fraction": args.fraction,
            "description": "Tests for linearity by comparing fit on central subset vs full data. Uses statsmodels.stats.diagnostic.linear_rainbow."
        }

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save to JSON with naming convention: {dataset}_rainbow.json
    output_file = os.path.join(args.output_dir, f"{dataset_name}_rainbow.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")


def write_output(output, output_dir, dataset_name, test_name):
    """Write output to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    # Save to JSON with naming convention: {dataset}_{test_name}.json
    output_file = os.path.join(output_dir, f"{dataset_name}_{test_name}.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
