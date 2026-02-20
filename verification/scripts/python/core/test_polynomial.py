# ============================================================================
# Polynomial Regression Test Reference Implementation (Python with statsmodels)
# ============================================================================
# This script generates reference values for polynomial regression using Python's
# statsmodels library. The test validates that the Rust implementation matches
# statsmodels' behavior for coefficient estimates and statistics.
#
# Source: statsmodels.regression.linear_model.OLS with polynomial features
# Reference: "Applied Linear Statistical Models" (Kutner et al.)
#
# Usage:
#   python test_polynomial.py --csv [csv_path] --output-dir [output_dir] [--degree DEGREE]
#   Args:
#     --csv        - Path to CSV file (first col = response, second col = predictor)
#                    Default: ../../datasets/csv/mtcars.csv
#     --output-dir - Path to output directory
#                    Default: ../../results/python
#     --degree     - Polynomial degree (default: 2)
# ============================================================================

import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import json
import sys
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """Helper for JSON serialization of numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def convert_categorical_to_numeric(data, dataset_name):
    """Convert categorical columns to numeric representations (0-based factor codes)."""
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

    if len(non_numeric_cols) > 0:
        print(f"INFO: Dataset '{dataset_name}' contains non-numeric columns: "
              f"{', '.join(non_numeric_cols)}")
        print("Converting categorical variables to numeric factor codes...")

        for col in non_numeric_cols:
            # First try to convert to numeric directly (handles numeric strings)
            numeric_vals = pd.to_numeric(data[col], errors='coerce')
            if not numeric_vals.isna().any():
                # All values converted successfully - use numeric values
                data[col] = numeric_vals
            else:
                # Contains non-numeric strings - convert to factor codes (0-based)
                codes, _ = pd.factorize(data[col])
                data[col] = codes.astype(float)
                print(f"  Column '{col}' encoded as 0-based factor codes")

    return data


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate polynomial regression reference values"
    )
    parser.add_argument(
        "--csv",
        default="../../datasets/csv/mtcars.csv",
        help="Path to CSV file (first col = response, second col = predictor)"
    )
    parser.add_argument(
        "--output-dir",
        default="../../results/python",
        help="Path to output directory"
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=2,
        help="Polynomial degree (default: 2)"
    )

    args = parser.parse_args()

    if args.degree < 1:
        print("ERROR: Polynomial degree must be at least 1")
        sys.exit(1)

    # Resolve paths - use absolute paths directly, resolve relative paths from cwd
    csv_path = Path(args.csv) if Path(args.csv).is_absolute() else Path.cwd() / args.csv
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else Path.cwd() / args.output_dir

    # Validate CSV path
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    # Extract dataset name from filename
    dataset_name = csv_path.stem
    degree = args.degree

    print(f"Running polynomial regression (degree={degree}) test on dataset: {dataset_name}")

    # Load data
    data = pd.read_csv(csv_path)
    data = convert_categorical_to_numeric(data, dataset_name)

    # Extract response (first column) and FIRST predictor (second column)
    # Polynomial regression uses a single predictor
    response_col = data.columns[0]
    predictor_col = data.columns[1]

    y = data[response_col].values
    x = data[predictor_col].values

    n = len(y)
    k = degree  # number of slope terms

    if n <= degree + 1:
        print(f"ERROR: Insufficient data: n={n}, degree={degree} (need n > degree+1)")
        sys.exit(1)

    # Build polynomial design matrix: [1, x, x^2, ..., x^degree]
    # This matches our implementation which uses raw (monomial) polynomial basis
    X_poly = np.column_stack([x ** d for d in range(degree + 1)])  # cols: 1, x, x^2, ...

    # Fit OLS model (intercept is already included in X_poly as x^0 = 1)
    model = sm.OLS(y, X_poly)
    results = model.fit()

    # Variable names matching our Rust implementation
    variable_names = ['Intercept'] + [f'x^{d}' for d in range(1, degree + 1)]

    # Coefficients and statistics
    coefficients = results.params.tolist()
    std_errors = results.bse.tolist()
    t_stats = results.tvalues.tolist()
    p_values = results.pvalues.tolist()

    # Model fit statistics
    df_residual = int(results.df_resid)
    df_model = degree

    r_squared = results.rsquared
    adj_r_squared = results.rsquared_adj
    f_statistic = results.fvalue
    f_p_value = results.f_pvalue

    # Residuals and error metrics
    residuals_arr = results.resid
    residuals = residuals_arr.tolist()
    fitted_values = results.fittedvalues.tolist()
    mse = results.mse_resid
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals_arr)))
    residual_std_error = float(np.sqrt(mse))

    # Model selection criteria
    log_likelihood = results.llf
    aic_val = results.aic
    bic_val = results.bic

    # Confidence intervals (95%)
    ci = results.conf_int(alpha=0.05)
    conf_int_lower = ci[:, 0].tolist()
    conf_int_upper = ci[:, 1].tolist()

    # Build formula string
    formula_str = f"{response_col} ~ poly({predictor_col}, {degree}, raw=True)"

    # Build result object
    result = {
        "test": "polynomial",
        "method": "statsmodels",
        "dataset": dataset_name,
        "formula": formula_str,
        "degree": degree,
        "n": n,
        "k": k,
        "df_residual": df_residual,
        "df_model": df_model,
        "variable_names": variable_names,
        "coefficients": coefficients,
        "std_errors": std_errors,
        "t_stats": t_stats,
        "p_values": p_values,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "f_statistic": f_statistic,
        "f_p_value": f_p_value,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "residual_std_error": residual_std_error,
        "log_likelihood": log_likelihood,
        "aic": aic_val,
        "bic": bic_val,
        "conf_int_lower": conf_int_lower,
        "conf_int_upper": conf_int_upper,
        "fitted_values": fitted_values,
        "residuals": residuals
    }

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write output
    output_file = output_dir / f"{dataset_name}_polynomial_degree{degree}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, cls=NumpyEncoder, indent=2)

    print(f"Wrote: {output_file.absolute()}")


if __name__ == "__main__":
    main()
