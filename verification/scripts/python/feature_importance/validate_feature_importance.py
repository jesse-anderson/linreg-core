# ============================================================================
# Feature Importance Validation - Python Reference Implementation
# ============================================================================
# This script computes feature importance metrics using NATIVE Python libraries
# (statsmodels, scikit-learn) to generate reference values for validation against
# the Rust implementation.
#
# Required packages:
#   pip install numpy pandas scipy statsmodels scikit-learn
# ============================================================================

import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict, Any

# Native Python libraries for reference implementation
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
from scipy import stats

# =============================================================================
# Data Helper Functions
# =============================================================================

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


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load a test dataset from CSV."""
    base_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'datasets', 'csv')

    datasets = {
        'mtcars': os.path.join(base_path, 'mtcars.csv'),
        'iris': os.path.join(base_path, 'iris.csv'),
        'faithful': os.path.join(base_path, 'faithful.csv'),
        'cars_stopping': os.path.join(base_path, 'cars_stopping.csv'),
    }

    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    df = pd.read_csv(datasets[dataset_name])
    df = convert_categorical_to_numeric(df, dataset_name)
    return df

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

    Reference: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
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

    Reference: https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html
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

    SHAP_i = coef_i * (x_i - mean(x_i))

    Reference: Lundberg & Lee (2017) - SHAP values for linear models
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

    # Create variable names
    variable_names = [f"X{j+1}" for j in range(n_features)]

    return {
        "variable_names": variable_names,
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
    n_permutations: int = 50,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Compute permutation importance using scikit-learn.

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
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

# =============================================================================
# Main Validation Runner
# =============================================================================

def main():
    """Run feature importance validation on all datasets using native Python libraries."""

    print("=" * 70)
    print("Feature Importance Validation - Python Reference Implementation")
    print("Using NATIVE libraries: statsmodels, scikit-learn, scipy")
    print("=" * 70)

    # Datasets to validate
    datasets = [
        "mtcars",
        "iris",
        "faithful",
        "cars_stopping",
    ]

    results = {}

    for dataset_name in datasets:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_name}")
        print('=' * 70)

        try:
            df = load_dataset(dataset_name)

            # Use first column as response, rest as predictors
            response = df.columns[0]
            predictors = [col for col in df.columns if col != response]

            print(f"Predictors: {predictors}")
            print(f"Response: {response}")
            print(f"Observations: {len(df)}")

            # Skip if too few predictors
            if len(predictors) < 1:
                print(f"Skipping {dataset_name}: insufficient predictors")
                continue

            dataset_results = {}

            # 1. Standardized Coefficients (statsmodels)
            print("\n[1/4] Standardized Coefficients...")
            try:
                std_coefs = validate_standardized_coefficients(df, predictors, response)
                dataset_results["standardized_coefficients"] = std_coefs
                print(f"   Variables: {std_coefs['variable_names']}")
                print(f"   Standardized Coefs: {[f'{x:.4f}' for x in std_coefs['standardized_coefficients']]}")
            except Exception as e:
                print(f"   Error: {e}")

            # 2. VIF Ranking (statsmodels)
            print("\n[2/4] VIF Ranking...")
            try:
                vif_result = validate_vif_ranking(df, predictors, response)
                dataset_results["vif_ranking"] = vif_result
                for v in vif_result['ranking'][:5]:  # Show first 5
                    print(f"   {v['variable']}: VIF={v['vif']:.2f} ({v['interpretation']})")
            except Exception as e:
                print(f"   Error: {e}")

            # 3. Linear SHAP (statsmodels)
            print("\n[3/4] Linear SHAP...")
            try:
                shap_result = validate_linear_shap(df, predictors, response)
                dataset_results["shap"] = shap_result
                print(f"   Mean |SHAP|: {[f'{x:.4f}' for x in shap_result['mean_abs_shap']]}")
            except Exception as e:
                print(f"   Error: {e}")

            # 4. Permutation Importance (scikit-learn)
            print("\n[4/4] Permutation Importance...")
            try:
                perm_result = validate_permutation_importance(df, predictors, response, n_permutations=10)
                dataset_results["permutation_importance"] = perm_result
                print(f"   Importance scores: {[f'{x:.4f}' for x in perm_result['importance']]}")
            except Exception as e:
                print(f"   Error: {e}")

            results[dataset_name] = dataset_results

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results to JSON
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'python', 'feature_importance')
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'feature_importance_reference.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_file}")
    print('=' * 70)

    # Print summary
    print("\nSUMMARY:")
    print("-------")
    for dataset_name, metrics in results.items():
        print(f"\n{dataset_name}:")
        if 'standardized_coefficients' in metrics:
            sc = metrics['standardized_coefficients']['standardized_coefficients']
            print(f"  Standardized Coeffs: {sc}")
        if 'vif_ranking' in metrics:
            v = metrics['vif_ranking']['vif_values']
            print(f"  VIF values: {v}")
        if 'shap' in metrics:
            s = metrics['shap']['mean_abs_shap']
            print(f"  Mean |SHAP|: {s}")
        if 'permutation_importance' in metrics:
            p = metrics['permutation_importance']['importance']
            print(f"  Permutation Importance: {p}")

if __name__ == "__main__":
    main()
