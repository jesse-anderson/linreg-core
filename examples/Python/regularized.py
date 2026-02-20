"""Regularized regression example: Ridge, Lasso, and Elastic Net.

Run with:
    pip install linreg-core
    python regularized.py

Demonstrates:
- Ridge regression (L2) for multicollinearity
- Lasso regression (L1) for variable selection
- Elastic Net (L1 + L2) as a blend
- Comparing all three on the same dataset
- Lambda sweep showing coefficient shrinkage path
"""

import linreg_core


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║         REGULARIZED REGRESSION — PYTHON BINDINGS                    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # ── Dataset ───────────────────────────────────────────────────────────────
    # y  = house price ($k)
    # x1 = square footage (hundreds)
    # x2 = number of bedrooms
    # x3 = age (years) — deliberately weak predictor
    y  = [245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1]
    x1 = [12.0, 18.0, 9.5, 24.0, 14.5, 20.0, 11.0, 28.0, 13.5, 16.5]
    x2 = [3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0]
    x3 = [15.0, 8.0, 30.0, 5.0, 12.0, 3.0, 25.0, 2.0, 18.0, 10.0]
    x_vars = [x1, x2, x3]

    print("Dataset: 10 houses, predictors: SqFt (hundreds), Bedrooms, Age")
    print()

    # ── 1. Ridge Regression ────────────────────────────────────────────────────
    print("━━━ 1. Ridge Regression (L2 — shrinks all, zeros none) ━━━━━━━━━━━━━━━")
    ridge = linreg_core.ridge_regression(y, x_vars, lambda_val=1.0, standardize=True)
    print(f"  Lambda:       1.0")
    print(f"  Intercept:    {ridge.intercept:.4f}")
    print(f"  SqFt:         {ridge.coefficients[0]:.4f}")
    print(f"  Bedrooms:     {ridge.coefficients[1]:.4f}")
    print(f"  Age:          {ridge.coefficients[2]:.4f}")
    print(f"  R²:           {ridge.r_squared:.4f}")
    print(f"  MSE:          {ridge.mse:.4f}")
    print(f"  Effective df: {ridge.effective_df:.4f}")
    print()
    print("  Ridge summary:")
    print(ridge.summary())
    print()

    # ── 2. Lasso Regression ────────────────────────────────────────────────────
    print("━━━ 2. Lasso Regression (L1 — zeros out weak predictors) ━━━━━━━━━━━━━")
    lasso = linreg_core.lasso_regression(
        y, x_vars, lambda_val=0.5, standardize=True, max_iter=10000, tol=1e-7
    )
    print(f"  Lambda:       0.5")
    print(f"  Intercept:    {lasso.intercept:.4f}")
    print(f"  SqFt:         {lasso.coefficients[0]:.4f}")
    print(f"  Bedrooms:     {lasso.coefficients[1]:.4f}")
    print(f"  Age:          {lasso.coefficients[2]:.4f}  {'<-- zeroed out' if abs(lasso.coefficients[2]) < 1e-6 else ''}")
    print(f"  R²:           {lasso.r_squared:.4f}")
    print(f"  MSE:          {lasso.mse:.4f}")
    print(f"  Non-zero:     {lasso.n_nonzero}/{len(lasso.coefficients)}")
    print(f"  Converged:    {lasso.converged}")
    print()

    # ── 3. Elastic Net ─────────────────────────────────────────────────────────
    print("━━━ 3. Elastic Net (alpha=0.5, equal L1+L2 blend) ━━━━━━━━━━━━━━━━━━━━")
    enet = linreg_core.elastic_net_regression(
        y, x_vars, lambda_val=0.5, alpha=0.5, standardize=True, max_iter=10000, tol=1e-7
    )
    print(f"  Lambda: 0.5  Alpha: 0.5")
    print(f"  Intercept:    {enet.intercept:.4f}")
    print(f"  SqFt:         {enet.coefficients[0]:.4f}")
    print(f"  Bedrooms:     {enet.coefficients[1]:.4f}")
    print(f"  Age:          {enet.coefficients[2]:.4f}")
    print(f"  R²:           {enet.r_squared:.4f}")
    print(f"  MSE:          {enet.mse:.4f}")
    print(f"  Non-zero:     {enet.n_nonzero}/{len(enet.coefficients)}")
    print(f"  Converged:    {enet.converged}")
    print()

    # ── 4. Lambda sweep (Lasso shrinkage path) ─────────────────────────────────
    print("━━━ 4. Lasso Shrinkage Path (lambda high -> low) ━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {'Lambda':>8}  {'SqFt':>10}  {'Bedrooms':>10}  {'Age':>10}  {'R²':>8}  {'Non-zero':>9}")
    print(f"  {'─'*64}")
    for lam in [50.0, 20.0, 8.0, 3.0, 1.0, 0.5, 0.1, 0.01]:
        fit = linreg_core.lasso_regression(
            y, x_vars, lambda_val=lam, standardize=True, max_iter=10000, tol=1e-7
        )
        print(f"  {lam:>8.2f}  {fit.coefficients[0]:>10.4f}  "
              f"{fit.coefficients[1]:>10.4f}  {fit.coefficients[2]:>10.4f}  "
              f"{fit.r_squared:>8.4f}  {fit.n_nonzero:>9}")
    print()
    print("  Note: As lambda decreases, coefficients grow from zero.")
    print("        Age enters last — it's the weakest predictor.")
    print()

    # ── 5. Side-by-side comparison ────────────────────────────────────────────
    print("━━━ 5. Method Comparison (lambda=0.5) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {'Method':<14}  {'R²':>8}  {'MSE':>10}  {'SqFt':>10}  {'Bedrooms':>10}  {'Age':>10}")
    print(f"  {'─'*68}")
    print(f"  {'Ridge':<14}  {ridge.r_squared:>8.4f}  {ridge.mse:>10.4f}  "
          f"{ridge.coefficients[0]:>10.4f}  {ridge.coefficients[1]:>10.4f}  "
          f"{ridge.coefficients[2]:>10.4f}")
    print(f"  {'Lasso':<14}  {lasso.r_squared:>8.4f}  {lasso.mse:>10.4f}  "
          f"{lasso.coefficients[0]:>10.4f}  {lasso.coefficients[1]:>10.4f}  "
          f"{lasso.coefficients[2]:>10.4f}")
    print(f"  {'Elastic Net':<14}  {enet.r_squared:>8.4f}  {enet.mse:>10.4f}  "
          f"{enet.coefficients[0]:>10.4f}  {enet.coefficients[1]:>10.4f}  "
          f"{enet.coefficients[2]:>10.4f}")


if __name__ == "__main__":
    main()
