# ============================================================================
# Ridge Regression Test Reference Implementation (Python with sklearn)
# ============================================================================
# This script generates reference values for ridge regression using Python's
# sklearn library. The test validates that the Rust implementation matches
# sklearn's Ridge behavior for coefficients and predictions.
#
# Source: sklearn.linear_model.Ridge
# Reference: "Regularization Paths for Generalized Linear Models via
#            Coordinate Descent" (Friedman, Hastie, Tibshirani 2010)
#
# Usage:
#   python test_ridge.py [csv_path] [output_dir] [lambda_count]
#   Args:
#     csv_path    - Path to CSV file (first col = response, rest = predictors)
#                   Default: ../../datasets/csv/mtcars.csv
#     output_dir  - Path to output directory
#                   Default: ../../results/python
#     lambda_count- Number of lambda values to generate
#                   Default: 20
# ============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import json
import os
import sys
from pathlib import Path

# Helper for JSON serialization of numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def generate_lambda_sequence(y, X, n_lambdas=20):
    """
    Generate a lambda sequence similar to glmnet's approach.
    glmnet starts from lambda_max (the smallest lambda where all coefficients
    are zero) and decreases logarithmically.
    """
    n, p = X.shape

    # Compute lambda_max: smallest lambda where all coefficients would be zero
    # For ridge, this is theoretically infinity, but we use a practical upper bound
    # based on the OLS solution scaled up
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression(fit_intercept=True)
    ols.fit(X, y)

    # Use cross-product of X and y as reference
    X_centered = X - X.mean(axis=0)
    y_centered = y - y.mean()

    # Maximum lambda that gives meaningful shrinkage
    lambda_max = np.max(np.abs(X_centered.T @ y_centered)) / n * 10

    # Logarithmic sequence from lambda_max to lambda_max * 1e-4
    lambda_sequence = np.logspace(np.log10(lambda_max), np.log10(lambda_max * 1e-4), n_lambdas)

    return lambda_sequence[::-1]  # Descending order like glmnet


def convert_categorical_to_numeric(data, dataset_name):
    """Convert categorical columns to numeric representations."""
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

    if len(non_numeric_cols) > 0:
        print(f"INFO: Dataset '{dataset_name}' contains non-numeric columns: "
              f"{', '.join(non_numeric_cols)}")
        print("Converting categorical variables to numeric representations...")

        for col in non_numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            # Fill NaN with mode
            if data[col].isna().any():
                mode_val = data[col].mode()
                if len(mode_val) > 0:
                    data[col].fillna(mode_val[0], inplace=True)

    return data


def main():
    # Parse command line arguments
    args = sys.argv[1:]

    # Set defaults
    default_csv = "../../datasets/csv/mtcars.csv"
    default_output = "../../results/python"
    default_lambda_count = 20

    csv_path_input = args[0] if len(args) >= 1 else default_csv
    output_dir_input = args[1] if len(args) >= 2 else default_output
    lambda_count = int(args[2]) if len(args) >= 3 else default_lambda_count

    # Resolve paths - use absolute paths directly, resolve relative paths from cwd
    csv_path = Path(csv_path_input) if Path(csv_path_input).is_absolute() else Path.cwd() / csv_path_input
    output_dir = Path(output_dir_input) if Path(output_dir_input).is_absolute() else Path.cwd() / output_dir_input

    # Validate CSV path
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    # Extract dataset name from filename
    dataset_name = csv_path.stem

    print(f"Running ridge regression test on dataset: {dataset_name}")

    # Load data
    data = pd.read_csv(csv_path)
    data = convert_categorical_to_numeric(data, dataset_name)

    # Extract response (first column) and predictors (remaining columns)
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values

    n, p = X.shape

    print(f"  n = {n} observations, p = {p} predictors")

    # Generate lambda sequence
    lambda_sequence = generate_lambda_sequence(y, X, lambda_count)

    # Store results for each lambda
    coefficients_list = []
    degrees_of_freedom = []

    # Standardize X for computing effective degrees of freedom
    scaler_X = StandardScaler()
    X_standardized = scaler_X.fit_transform(X)

    for lam in lambda_sequence:
        # Fit ridge regression
        # sklearn uses alpha = lambda / n (approximately)
        # We need to match glmnet's objective function
        # glmnet: (1/2n) * ||y - Xb||^2 + (lambda/2) * ||b||^2
        # sklearn: ||y - Xb||^2 + alpha * ||b||^2
        # So alpha = lambda
        ridge = Ridge(alpha=lam, fit_intercept=True,
                      solver='auto', random_state=42)

        # sklearn expects standardized data internally for Ridge
        # We'll manually standardize to match glmnet's behavior
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0, ddof=0)  # ddof=0 for population std (glmnet uses this)

        # Handle constant columns
        X_std_safe = np.where(X_std < 1e-10, 1.0, X_std)

        X_scaled = (X - X_mean) / X_std_safe

        # Fit on standardized data
        ridge.fit(X_scaled, y)

        # Get coefficients on original scale
        # coef_original = coef_scaled / std
        coef_scaled = ridge.coef_
        intercept = ridge.intercept_ - np.sum(coef_scaled * X_mean / X_std_safe)
        coef_original = coef_scaled / X_std_safe

        # Store coefficients (intercept first, then coefficients)
        coefs_with_intercept = np.concatenate([[intercept], coef_original])
        coefficients_list.append(coefs_with_intercept.tolist())

        # Compute effective degrees of freedom for ridge
        # df = trace(X (X'X + lambda*I)^-1 X')
        # For standardized X: df = sum(s_i^2 / (s_i^2 + lambda)) where s_i are singular values
        U, s, Vt = np.linalg.svd(X_standardized, full_matrices=False)
        df_lam = np.sum(s**2 / (s**2 + lam))
        degrees_of_freedom.append(df_lam)

    # Get predictions at a few representative lambdas
    test_indices = [0, lambda_count // 2, lambda_count - 1]
    test_lambdas = lambda_sequence[test_indices].tolist()

    # Create test data (use first min(5, n) rows for prediction tests)
    n_test = min(5, n)
    X_test = X[:n_test, :]

    predictions = []
    for idx in test_indices:
        lam = lambda_sequence[idx]

        ridge = Ridge(alpha=lam, fit_intercept=True, solver='auto', random_state=42)
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0, ddof=0)
        X_std_safe = np.where(X_std < 1e-10, 1.0, X_std)
        X_scaled = (X - X_mean) / X_std_safe
        ridge.fit(X_scaled, y)

        # Predict on test data
        X_test_scaled = (X_test - X_mean) / X_std_safe
        pred = ridge.predict(X_test_scaled)
        predictions.append(pred.tolist())

    # Compute fitted values at the final lambda
    lam_final = lambda_sequence[-1]
    ridge_final = Ridge(alpha=lam_final, fit_intercept=True, solver='auto', random_state=42)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=0)
    X_std_safe = np.where(X_std < 1e-10, 1.0, X_std)
    X_scaled = (X - X_mean) / X_std_safe
    ridge_final.fit(X_scaled, y)

    fitted_values = ridge_final.predict(X_scaled).tolist()
    residuals = (y - np.array(fitted_values)).tolist()

    # Build result object
    result = {
        "test": "ridge",
        "method": "sklearn",
        "alpha": 0,
        "n": n,
        "p": p,
        "lambda_sequence": lambda_sequence.tolist(),
        "coefficients": coefficients_list,
        "degrees_of_freedom": degrees_of_freedom,
        "test_lambdas": test_lambdas,
        "test_predictions": predictions,
        "fitted_values": fitted_values,
        "residuals": residuals,
        "sklearn_version": sklearn.__version__
    }

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write output
    output_file = output_dir / f"{dataset_name}_ridge.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, cls=NumpyEncoder, indent=2)

    print(f"Wrote: {output_file.absolute()}")


if __name__ == "__main__":
    import sklearn  # Import here for version check
    main()
