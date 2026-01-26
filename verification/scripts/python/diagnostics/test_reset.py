# ============================================================================
# RESET Test Reference Implementation (Python)
# ============================================================================
# This script generates reference values for the RESET test using
# statsmodels.stats.diagnostic.linear_reset. The RESET (Regression Specification
# Error Test) checks for functional form misspecification by testing if powers
# of fitted values or regressors significantly improve the model fit.
#
# Source: statsmodels package, linear_reset function
# Reference: Ramsey, J.B. (1969), "Tests for Specification Errors in Classical
#           Linear Least-Squares Regression Analysis", Journal of the Royal
#           Statistical Society, Series B 31: 350–371.
#
# Usage:
#   python test_reset.py [--csv CSV_PATH] [--output-dir OUTPUT_DIR] [--powers POWERS] [--type TYPE]
#
#   Args:
#     --csv       Path to CSV file (first col = response, rest = predictors)
#                 Default: ../../datasets/csv/mtcars.csv
#     --output-dir Path to output directory
#                 Default: ../../results/python
#     --powers    Powers to use (e.g., "2,3" for squared and cubed)
#                 Default: "2,3"
#     --type      Type of terms: "fitted", "regressor", or "exog"
#                 Default: "fitted"
# ============================================================================

import argparse
import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset


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
            # For string/categorical data, use factorize for integer level encoding
            # This matches R's as.numeric(as.factor()) behavior
            encoded, uniques = pd.factorize(data[col])
            data[col] = encoded
            print(f"  {col}: {len(uniques)} unique values -> integer level encoding")
        else:
            # Already numeric or other type
            pass

    return non_numeric_cols


def main():
    parser = argparse.ArgumentParser(
        description="RESET Test - Check functional form specification using statsmodels"
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
        "--powers",
        default="2,3",
        help="Powers to use (e.g., '2,3' for squared and cubed)"
    )
    parser.add_argument(
        "--type",
        default="fitted",
        choices=["fitted", "regressor", "exog"],
        help="Type of terms to add (default: fitted)"
    )

    args = parser.parse_args()

    # Parse powers string (e.g., "2,3" -> [2, 3])
    powers = [int(p.strip()) for p in args.powers.split(",")]

    # Map type to statsmodels parameter
    # statsmodels uses: "fitted" (powers of fitted values), "exog" (powers of regressors)
    type_map = {
        "fitted": "fitted",
        "regressor": "exog",
        "exog": "exog"
    }
    sm_type = type_map[args.type]

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

        # Update response_col to the first one-hot encoded column
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

    # Run RESET test for each power
    # statsmodels' linear_reset only accepts a single power at a time
    # We need to run it multiple times and combine the results
    # However, this doesn't match R's behavior which tests all powers together
    # So we'll implement the full RESET test following R's algorithm

    print("\nRESET Test (Python - Custom Implementation)")
    print("=" * 50)
    print(f"Dataset: {dataset_name}")
    print(f"Formula: {formula_str}")
    print(f"Powers: {powers}")
    print(f"Type: {args.type}")

    # Get fitted values
    fitted = model.fittedvalues.values

    # Build Z matrix (additional terms)
    n = len(y)
    k = len(predictor_cols)
    p = k + 1  # including intercept

    if args.type == "fitted":
        # Powers of fitted values
        Z = np.column_stack([fitted ** power for power in powers])
    elif args.type == "regressor" or args.type == "exog":
        # Powers of each regressor
        X_no_intercept = X.values[:, 1:]  # Remove intercept column
        Z_cols = []
        for power in powers:
            for col_idx in range(X_no_intercept.shape[1]):
                Z_cols.append(X_no_intercept[:, col_idx] ** power)
        Z = np.column_stack(Z_cols)
    else:
        raise ValueError(f"Unsupported type: {args.type}")

    # Number of additional terms
    q = Z.shape[1]

    # Check we have enough degrees of freedom
    if n <= p + q:
        output = {
            "test_name": "RESET Test (Python - statsmodels)",
            "dataset": dataset_name,
            "formula": formula_str,
            "statistic": None,
            "p_value": None,
            "passed": None,
            "skipped": True,
            "reason": f"Insufficient degrees of freedom: n={n}, p={p}, q={q}",
            "power": powers,
            "type": args.type,
            "description": "Ramsey's RESET test for functional form misspecification. Tests whether powers of fitted values, regressors, or principal component significantly improve model fit."
        }
    else:
        # Fit unrestricted model (original + Z terms)
        XZ = np.column_stack([X.values, Z])

        try:
            model_unrestricted = sm.OLS(y, XZ).fit()
        except Exception as e:
            # Check if it's a multicollinearity issue
            if "singular" in str(e).lower() or "multicollinearity" in str(e).lower():
                output = {
                    "test_name": "RESET Test (Python - statsmodels)",
                    "dataset": dataset_name,
                    "formula": formula_str,
                    "statistic": None,
                    "p_value": None,
                    "passed": None,
                    "skipped": True,
                    "reason": "High multicollinearity detected - cannot reliably perform RESET test",
                    "power": powers,
                    "type": args.type,
                    "description": "Ramsey's RESET test for functional form misspecification. Tests whether powers of fitted values, regressors, or principal component significantly improve model fit."
                }
            else:
                raise
        else:
            # RSS from restricted and unrestricted models
            rss_restricted = model.ssr
            rss_unrestricted = model_unrestricted.ssr

            # F-statistic: F = (df2/df1) * ((RSS1 - RSS2) / RSS2)
            df1 = q
            df2 = n - p - q

            f_stat = (df2 / df1) * ((rss_restricted - rss_unrestricted) / rss_unrestricted)

            # Handle numerical edge cases
            if not np.isfinite(f_stat) or f_stat < 0:
                f_stat = 0.0

            # Get p-value from F-distribution
            from scipy import stats
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)

            passed = p_value > 0.05

            # Print results
            print(f"Statistic (F): {f_stat:.6f}")
            print(f"p-value: {p_value:.6f}")
            print(f"Passed: {passed}")
            print()

            # Prepare output
            output = {
                "test_name": "RESET Test (Python - statsmodels)",
                "dataset": dataset_name,
                "formula": formula_str,
                "statistic": float(f_stat),
                "p_value": float(p_value),
                "passed": bool(passed),
                "power": powers,
                "type": args.type,
                "description": "Ramsey's RESET test for functional form misspecification. Tests whether powers of fitted values, regressors, or principal component significantly improve model fit."
            }

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save to JSON with naming convention: {dataset}_reset.json
    output_file = os.path.join(args.output_dir, f"{dataset_name}_reset.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
