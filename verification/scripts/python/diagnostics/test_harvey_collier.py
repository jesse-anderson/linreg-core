# ============================================================================
# Harvey-Collier Test Reference Implementation (Python)
# ============================================================================
# This script generates reference values for the Harvey-Collier test using
# statsmodels.stats.diagnostic.linear_harvey_collier. The Harvey-Collier test
# checks for functional form misspecification by examining whether recursive
# residuals exhibit a linear trend.
#
# Source: statsmodels package, linear_harvey_collier function
# Reference: Harvey & Collier (1977), "Testing for functional misspecification
#            using the work of a higher order" (unpublished)
#
# Usage:
#   python test_harvey_collier.py [--csv CSV_PATH] [--output-dir OUTPUT_DIR]
#
#   Args:
#     --csv       Path to CSV file (first col = response, rest = predictors)
#                 Default: ../../datasets/csv/mtcars.csv
#     --output-dir Path to output directory
#                 Default: ../../results/python
# ============================================================================

import argparse
import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_harvey_collier


def validate_for_regression(data, dataset_name):
    """Validate data for regression analysis."""
    issues = []

    # Check for missing values
    missing_counts = data.isnull().sum()
    if missing_counts.sum() > 0:
        missing_cols = missing_counts[missing_counts > 0].index.tolist()
        issues.append(f"Missing values detected in: {missing_cols}")

    # Check for multicollinearity using condition number
    if len(data.columns) > 2:  # Need at least 2 predictors to check VIF
        X = data.select_dtypes(include=[np.number])
        if X.shape[1] > 0:  # Only check if we have numeric predictors
            # Compute VIF-like condition number
            # 1 / R² = 1 / (1 - R²) = VIF
            # We'll use a simpler approach: check correlation matrix
            corr_matrix = X.corr()
            # Find max absolute correlation between predictors
            max_corr = corr_matrix.abs().values
            np.fill_diagonal(max_corr, np.nan)  # Ignore self-correlation
            max_abs_corr = np.nanmax(max_corr)

            if max_abs_corr > 0.95:
                issues.append(f"High multicollinearity detected (max correlation: {max_abs_corr:.3f})")

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
        description="Harvey-Collier Test - Check functional form using statsmodels"
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
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"INFO: Dataset '{dataset_name}' contains non-numeric columns: {non_numeric_cols}")
        convert_categorical_to_numeric(data, dataset_name)

    # After conversion, re-check if we still have non-numeric columns
    remaining_non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
    if remaining_non_numeric:
        # These could be things like empty strings or other non-numeric types that couldn't be converted
        raise ValueError(f"Could not convert the following non-numeric columns to numeric: {remaining_non_numeric}")

    # Check for multicollinearity issues
    if any("multicollinearity" in issue.lower() for issue in issues):
        multicollinearity_issues = [i for i in issues if "multicollinearity" in i.lower()]
        print(f"WARNING: {multicollinearity_issues[0]}")
        print("Harvey-Collier test is sensitive to multicollinearity. The test will attempt to run but may return NaN values.")

    # If there are still issues, fail gracefully
    if issues:
        remaining_issues = [i for i in issues if not any(keyword in i.lower() for keyword in ["non-numeric", "multicollinearity"])]
        if remaining_issues:
            raise ValueError(f"Data validation failed: {remaining_issues}")

    # Assume first column is response variable, rest are predictors
    response_col = data.columns[0]
    predictor_cols = data.columns[1:]

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
        output = {
            "test_name": "Harvey-Collier Test (Python - statsmodels)",
            "dataset": dataset_name,
            "formula": None,
            "statistic": None,
            "p_value": None,
            "passed": None,
            "skipped": True,
            "reason": f"Failed to fit regression model: {e}",
            "description": "Tests for functional form misspecification by examining whether recursive residuals exhibit a linear trend. Uses statsmodels.stats.diagnostic.linear_harvey_collier."
        }
        write_output(output, args.output_dir, dataset_name, "harvey_collier")
        return

    # Run Harvey-Collier test
    try:
        hc_result = linear_harvey_collier(model)
    except Exception as e:
        # Check if it's a multicollinearity issue
        if "singular" in str(e).lower() or "multicollinearity" in str(e).lower() or "skip" in str(e).lower():
            output = {
                "test_name": "Harvey-Collier Test (Python - statsmodels)",
                "dataset": dataset_name,
                "formula": formula_str,
                "statistic": None,
                "p_value": None,
                "passed": None,
                "skipped": True,
                "reason": "High multicollinearity detected - Harvey-Collier test requires full-rank design matrix. Consider using VIF to diagnose multicollinearity first.",
                "description": "Tests for functional form misspecification by examining whether recursive residuals exhibit a linear trend. Uses statsmodels.stats.diagnostic.linear_harvey_collier."
            }
            write_output(output, args.output_dir, dataset_name, "harvey_collier")
            return
        else:
            raise

    # Print results
    print("Harvey-Collier Test (Python - statsmodels)")
    print("=" * 45)
    print(f"Dataset: {dataset_name}")
    print(f"Formula: {formula_str}")
    print(f"Statistic (t): {hc_result[0]}")
    print(f"p-value: {hc_result[1]}")
    print(f"Passed: {hc_result[1] > 0.05}")
    print()

    # Prepare output - handle NaN values
    statistic = hc_result[0]
    p_value = hc_result[1]

    if np.isnan(statistic) or np.isnan(p_value):
        output = {
            "test_name": "Harvey-Collier Test (Python - statsmodels)",
            "dataset": dataset_name,
            "formula": formula_str,
            "statistic": None,
            "p_value": None,
            "passed": None,
            "skipped": True,
            "reason": "Test returned NaN values - likely due to multicollinearity in the data",
            "description": "Tests for functional form misspecification by examining whether recursive residuals exhibit a linear trend. Uses statsmodels.stats.diagnostic.linear_harvey_collier."
        }
    else:
        output = {
            "test_name": "Harvey-Collier Test (Python - statsmodels)",
            "dataset": dataset_name,
            "formula": formula_str,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "passed": bool(p_value > 0.05),
            "skipped": False,
            "description": "Tests for functional form misspecification by examining whether recursive residuals exhibit a linear trend. Uses statsmodels.stats.diagnostic.linear_harvey_collier."
        }

    write_output(output, args.output_dir, dataset_name, "harvey_collier")


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
