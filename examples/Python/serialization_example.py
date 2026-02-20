"""Model serialization example — save and load trained regression models.

Run with:
    pip install linreg-core
    python serialization_example.py

Demonstrates:
- Saving OLS, Ridge, Lasso, Elastic Net, WLS, and LOESS models to JSON
- Loading them back and verifying coefficients are preserved
- Inspecting saved metadata (model type, version, name, timestamp)
- Error handling for wrong-type loads and missing files
"""

import json
import os
import linreg_core


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║         MODEL SERIALIZATION — PYTHON BINDINGS                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # Shared sample data
    y  = [2.5, 3.7, 4.2, 5.1, 6.3, 7.0, 8.1, 9.2]
    x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    x2 = [1.5, 2.1, 3.2, 3.9, 5.1, 6.2, 7.0, 8.1]
    x_vars = [x1, x2]
    names  = ["Intercept", "X1", "X2"]

    paths = []

    # ── 1. OLS ─────────────────────────────────────────────────────────────────
    print("━━━ 1. OLS Regression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    ols = linreg_core.ols_regression(y, x_vars, names)
    print(f"  R²: {ols.r_squared:.4f}   Coefficients: {[round(c,4) for c in ols.coefficients]}")

    meta = linreg_core.save_model(ols, "ols_model.json", name="My OLS Model")
    print(f"  Saved  -> {meta['path']}  (type: {meta['model_type']})")
    paths.append("ols_model.json")

    ols2 = linreg_core.load_model("ols_model.json")
    print(f"  Loaded -> R²: {ols2.r_squared:.4f}   "
          f"Coefficients: {[round(c,4) for c in ols2.coefficients]}")
    print()

    # ── 2. Ridge ───────────────────────────────────────────────────────────────
    print("━━━ 2. Ridge Regression (λ=1.0) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    ridge = linreg_core.ridge_regression(y, x_vars, lambda_val=1.0, standardize=True)
    print(f"  Intercept: {ridge.intercept:.4f}   "
          f"Coefficients: {[round(c,4) for c in ridge.coefficients]}")

    linreg_core.save_model(ridge, "ridge_model.json")
    print(f"  Saved  -> ridge_model.json")
    paths.append("ridge_model.json")

    ridge2 = linreg_core.load_model("ridge_model.json")
    print(f"  Loaded -> Intercept: {ridge2.intercept:.4f}   "
          f"Coefficients: {[round(c,4) for c in ridge2.coefficients]}")
    print()

    # ── 3. Lasso ───────────────────────────────────────────────────────────────
    print("━━━ 3. Lasso Regression (λ=0.1) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lasso = linreg_core.lasso_regression(
        y, x_vars, lambda_val=0.1, standardize=True, max_iter=10000, tol=1e-7
    )
    print(f"  Intercept: {lasso.intercept:.4f}   Non-zero: {lasso.n_nonzero}   "
          f"Converged: {lasso.converged}")

    linreg_core.save_model(lasso, "lasso_model.json")
    print(f"  Saved  -> lasso_model.json")
    paths.append("lasso_model.json")

    lasso2 = linreg_core.load_model("lasso_model.json")
    print(f"  Loaded -> Intercept: {lasso2.intercept:.4f}   "
          f"Coefficients: {[round(c,4) for c in lasso2.coefficients]}")
    print()

    # ── 4. Elastic Net ─────────────────────────────────────────────────────────
    print("━━━ 4. Elastic Net (λ=0.1, α=0.5) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    enet = linreg_core.elastic_net_regression(
        y, x_vars, lambda_val=0.1, alpha=0.5, standardize=True, max_iter=10000, tol=1e-7
    )
    print(f"  Intercept: {enet.intercept:.4f}   α={enet.alpha}   Non-zero: {enet.n_nonzero}")

    linreg_core.save_model(enet, "enet_model.json")
    print(f"  Saved  -> enet_model.json")
    paths.append("enet_model.json")

    enet2 = linreg_core.load_model("enet_model.json")
    print(f"  Loaded -> Intercept: {enet2.intercept:.4f}   "
          f"Coefficients: {[round(c,4) for c in enet2.coefficients]}")
    print()

    # ── 5. WLS ─────────────────────────────────────────────────────────────────
    print("━━━ 5. WLS Regression (equal weights) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    weights = [1.0] * len(y)
    wls = linreg_core.wls_regression(y, x_vars, weights)
    print(f"  R²: {wls.r_squared:.4f}   "
          f"Coefficients: {[round(c,4) for c in wls.coefficients]}")

    linreg_core.save_model(wls, "wls_model.json")
    print(f"  Saved  -> wls_model.json")
    paths.append("wls_model.json")

    wls2 = linreg_core.load_model("wls_model.json")
    print(f"  Loaded -> R²: {wls2.r_squared:.4f}   "
          f"Coefficients: {[round(c,4) for c in wls2.coefficients]}")
    print()

    # ── 6. LOESS ───────────────────────────────────────────────────────────────
    print("━━━ 6. LOESS Regression (span=0.75, degree=2) ━━━━━━━━━━━━━━━━━━━━━━")
    loess = linreg_core.loess_fit(y, [x1], span=0.75, degree=2, robust_iterations=2)
    print(f"  Span: {loess.span}   Degree: {loess.degree}   "
          f"First 3 fitted: {[round(v,4) for v in loess.fitted[:3]]}")

    linreg_core.save_model(loess, "loess_model.json")
    print(f"  Saved  -> loess_model.json")
    paths.append("loess_model.json")

    loess2 = linreg_core.load_model("loess_model.json")
    print(f"  Loaded -> Span: {loess2.span}   "
          f"First 3 fitted: {[round(v,4) for v in loess2.fitted[:3]]}")
    print()

    # ── 7. Metadata Inspection ─────────────────────────────────────────────────
    print("━━━ 7. Metadata Inspection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    with open("ols_model.json") as f:
        raw = json.load(f)
    meta = raw["metadata"]
    print(f"  Model type:      {meta['model_type']}")
    print(f"  Format version:  {meta['format_version']}")
    print(f"  Library version: {meta['library_version']}")
    print(f"  Created at:      {meta['created_at']}")
    print(f"  Name:            {meta.get('name', '(none)')}")
    print()

    # ── 8. Error Handling ──────────────────────────────────────────────────────
    print("━━━ 8. Error Handling ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    try:
        linreg_core.load_model("nonexistent.json")
    except OSError as e:
        print(f"  Missing file error (expected): {type(e).__name__}")

    print()

    # ── Cleanup ────────────────────────────────────────────────────────────────
    print("━━━ Cleanup ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for path in paths:
        os.remove(path)
        print(f"  Removed: {path}")
    print()
    print("  Done.")


if __name__ == "__main__":
    main()
