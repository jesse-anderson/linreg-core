# ============================================================================
# Feature Importance Test Reference Implementation (Python with statsmodels)
# ============================================================================
# This script generates reference values for feature importance using Python's
# statsmodels and scikit-learn libraries. The test validates that the Rust
# implementation matches Python's behavior for feature importance metrics.
#
# Required packages:
#   pip install numpy pandas scipy statsmodels scikit-learn
#
# Usage:
#   python test_feature_importance.py --csv [csv_path] --output-dir [output_dir]
#   Args:
#     --csv        - Path to CSV file (first col = response, rest = predictors)
#                    Default: ../../datasets/csv/mtcars.csv
#     --output-dir - Path to output directory
#                    Default: ../../results/python
# ============================================================================

import argparse
import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Native Python libraries for reference implementation
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
from scipy import stats


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
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


# =============================================================================
# Standardized Coefficients (using statsmodels)
# =============================================================================

def validate_standardized_coefficients(
    df: pd.DataFrame,
    predictors: List[str],
    response: str
) -> Dict[str, Any]:
    """
    Compute standardized coefficients using statsmodels.
    """
    X = df[predictors].values
    y = df[response].values

    # Fit OLS model
    X_with_const = sm.add_constant(X)
    model = OLS(y, X_with_const).fit()

    # Get coefficients (excluding intercept)
    coefficients = model.params[1:]  # Skip intercept

    # Compute standard deviations (using scipy.stats for compatibility)
    x_stds = stats.tstd(X, ddof=X.shape[0] - 1, axis=0)
    y_std = stats.tstd(y, ddof=len(y) - 1)

    # Standardized coefficients: beta* = beta * (sigma_x / sigma_y)
    std_coefs = coefficients * (x_stds / y_std)

    result = {
        "variable_names": predictors,
        "standardized_coefficients": std_coefs.tolist(),
        "y_std": float(y_std),
        "raw_coefficients": coefficients.tolist()
    }

    return result


# =============================================================================
# VIF Ranking (using statsmodels)
# =============================================================================

def validate_vif_ranking(
    df: pd.DataFrame,
    predictors: List[str],
    response: str
) -> Dict[str, Any]:
    """
    Compute VIF values using statsmodels.
    """
    X = df[predictors].values
    X_with_const = sm.add_constant(X)

    vif_data = []
    for i, var_name in enumerate(predictors):
        # VIF calculation using statsmodels
        vif_val = variance_inflation_factor(X_with_const, i + 1)

        # Handle edge cases
        if np.isinf(vif_val):
            rsquared = 1.0
        elif vif_val > 0:
            rsquared = max(0.0, 1.0 - 1.0 / vif_val)
        else:
            rsquared = 0.0

        # Determine interpretation
        if vif_val < 5:
            interpretation = "Low multicollinearity"
        elif vif_val < 10:
            interpretation = "Moderate multicollinearity"
        else:
            interpretation = "High multicollinearity"

        vif_data.append({
            "variable": var_name,
            "vif": float(vif_val) if np.isfinite(vif_val) else float('inf'),
            "rsquared": rsquared,
            "interpretation": interpretation
        })

    # Sort by VIF (ascending = least redundant first)
    vif_sorted = sorted(vif_data, key=lambda x: x['vif'] if x['vif'] != float('inf') else float('inf'))

    return {
        "variable_names": [v['variable'] for v in vif_sorted],
        "vif_values": [v['vif'] for v in vif_sorted],
        "ranking": vif_sorted
    }


# =============================================================================
# Linear SHAP Values (exact, closed-form)
# =============================================================================

def validate_linear_shap(
    df: pd.DataFrame,
    predictors: List[str],
    response: str
) -> Dict[str, Any]:
    """
    Compute exact SHAP values for linear regression using statsmodels.
    """
    X = df[predictors].values
    y = df[response].values

    # Fit OLS model
    X_with_const = sm.add_constant(X)
    model = OLS(y, X_with_const).fit()

    # Get coefficients
    coefficients = model.params[1:]  # Skip intercept
    intercept = model.params[0]

    # Compute means of predictors
    X_means = X.mean(axis=0)

    n_obs = len(y)
    n_features = len(predictors)

    # Compute SHAP values matrix
    shap_values = []
    for i in range(n_obs):
        obs_shap = []
        for j in range(n_features):
            shap_val = coefficients[j] * (X[i, j] - X_means[j])
            obs_shap.append(shap_val)
        shap_values.append(obs_shap)

    # Compute mean absolute SHAP per feature
    shap_array = np.array(shap_values)
    mean_abs_shap = np.abs(shap_array).mean(axis=0).tolist()

    return {
        "variable_names": predictors,
        "shap_values": shap_values,
        "base_value": float(intercept),
        "mean_abs_shap": mean_abs_shap
    }


# =============================================================================
# Permutation Importance (using scikit-learn)
# =============================================================================

def validate_permutation_importance(
    df: pd.DataFrame,
    predictors: List[str],
    response: str,
    n_permutations: int = 10,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Compute permutation importance using scikit-learn.
    """
    np.random.seed(random_seed)

    X = df[predictors].values
    y = df[response].values

    # Fit baseline model
    from sklearn.linear_model import LinearRegression
    from sklearn.base import clone

    model = LinearRegression()
    model.fit(X, y)
    baseline_r2 = r2_score(y, model.predict(X))

    n_obs, n_features = X.shape
    importance_scores = []

    for j in range(n_features):
        perm_r2_sum = 0.0

        for _ in range(n_permutations):
            X_permuted = X.copy()

            # Shuffle column j
            perm_col = X_permuted[:, j].copy()
            np.random.shuffle(perm_col)
            X_permuted[:, j] = perm_col

            # Fit on permuted data and compute R²
            perm_model = clone(model)
            perm_model.fit(X_permuted, y)
            perm_r2 = r2_score(y, perm_model.predict(X_permuted))

            perm_r2_sum += baseline_r2 - perm_r2

        importance_scores.append(perm_r2_sum / n_permutations)

    return {
        "variable_names": predictors,
        "importance": importance_scores,
        "baseline_score": float(baseline_r2),
        "n_permutations": n_permutations,
        "seed": random_seed
    }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate feature importance reference values"
    )
    parser.add_argument(
        "--csv",
        default="../../datasets/csv/mtcars.csv",
        help="Path to CSV file (first col = response, rest = predictors)"
    )
    parser.add_argument(
        "--output-dir",
        default="../../results/python",
        help="Path to output directory"
    )

    args = parser.parse_args()

    # Resolve paths
    csv_path = Path(args.csv) if Path(args.csv).is_absolute() else Path.cwd() / args.csv
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else Path.cwd() / args.output_dir

    # Validate CSV path
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    # Extract dataset name from filename
    dataset_name = csv_path.stem

    print(f"Running feature importance test on dataset: {dataset_name}")

    # Load data
    data = pd.read_csv(csv_path)
    data = convert_categorical_to_numeric(data, dataset_name)

    # Use first column as response, rest as predictors
    response = data.columns[0]
    predictors = [col for col in data.columns if col != response]

    print(f"Predictors: {predictors}")
    print(f"Response: {response}")
    print(f"Observations: {len(data)}")

    # Skip if too few predictors
    if len(predictors) < 1:
        print(f"Skipping {dataset_name}: insufficient predictors")
        sys.exit(1)

    n_features = len(predictors)

    # Single predictor case - skip VIF (requires at least 2 predictors)
    has_vif = n_features >= 2

    result = {
        "test": "feature_importance",
        "method": "statsmodels",
        "dataset": dataset_name,
        "n": len(data),
        "k": n_features,
        "variable_names": predictors,
        "response": response
    }

    # 1. Standardized Coefficients
    print("\n[1/4] Standardized Coefficients...")
    try:
        std_coefs = validate_standardized_coefficients(data, predictors, response)
        result["standardized_coefficients"] = std_coefs
        print(f"   Variables: {std_coefs['variable_names']}")
        print(f"   Standardized Coefs: {[f'{x:.4f}' for x in std_coefs['standardized_coefficients']]}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()

    # 2. VIF Ranking (skip for single predictor)
    if has_vif:
        print("\n[2/4] VIF Ranking...")
        try:
            vif_result = validate_vif_ranking(data, predictors, response)
            result["vif_ranking"] = vif_result
            for v in vif_result['ranking'][:5]:
                print(f"   {v['variable']}: VIF={v['vif']:.2f} ({v['interpretation']})")
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[2/4] VIF Ranking... Skipped (only 1 predictor)")

    # 3. Linear SHAP
    print("\n[3/4] Linear SHAP...")
    try:
        shap_result = validate_linear_shap(data, predictors, response)
        result["shap"] = shap_result
        print(f"   Mean |SHAP|: {[f'{x:.4f}' for x in shap_result['mean_abs_shap']]}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()

    # 4. Permutation Importance
    print("\n[4/4] Permutation Importance...")
    try:
        perm_result = validate_permutation_importance(data, predictors, response, n_permutations=10)
        result["permutation_importance"] = perm_result
        print(f"   Importance scores: {[f'{x:.4f}' for x in perm_result['importance']]}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write output
    output_file = output_dir / f"{dataset_name}_feature_importance.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, cls=NumpyEncoder, indent=2)

    print(f"\nWrote: {output_file.absolute()}")


if __name__ == "__main__":
    main()
