"""K-Fold Cross Validation example.

Run with:
    pip install linreg-core
    python cross_validation.py

Demonstrates:
- Native kfold_cv_ols / kfold_cv_ridge / kfold_cv_lasso / kfold_cv_elastic_net
- Lambda selection for regularized regression via CV
- Alpha selection for Elastic Net via CV
- Per-fold FoldResult objects (fold_index, train_size, test_size, rmse, r_squared, ...)
- Coefficient stability via fold_coefficients
- Reproducibility using a fixed random seed
"""

import math
import linreg_core


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║              K-FOLD CROSS VALIDATION — PYTHON BINDINGS             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # Housing price dataset (25 observations)
    y = [
        245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1,
        445.8, 167.9, 367.4, 289.6, 198.2, 478.5, 256.3, 334.7, 178.5, 398.9,
        223.4, 312.5, 156.8, 423.7, 267.9,
    ]
    sqft = [
        1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0,
        2200.0, 900.0, 1950.0, 1500.0, 1050.0, 2600.0, 1300.0, 1850.0, 1000.0, 2100.0,
        1250.0, 1700.0, 850.0, 2350.0, 1400.0,
    ]
    bedrooms = [
        3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0,
        4.0, 2.0, 4.0, 3.0, 2.0, 5.0, 3.0, 4.0, 2.0, 4.0,
        3.0, 3.0, 2.0, 4.0, 3.0,
    ]
    age = [
        15.0, 10.0, 25.0, 5.0, 8.0, 12.0, 20.0, 2.0, 18.0, 7.0,
        3.0, 30.0, 6.0, 14.0, 22.0, 1.0, 16.0, 9.0, 28.0, 4.0,
        19.0, 11.0, 35.0, 3.0, 13.0,
    ]
    x_vars = [sqft, bedrooms, age]
    names  = ["Intercept", "SqFt", "Bedrooms", "Age"]

    # ── 1. OLS Cross Validation ───────────────────────────────────────────────
    print("━━━ 1. OLS — 5-Fold Cross Validation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    cv = linreg_core.kfold_cv_ols(y, x_vars, names, n_folds=5, shuffle=True, seed=42)

    print(f"  Mean RMSE:      {cv.mean_rmse:.4f} (±{cv.std_rmse:.4f})")
    print(f"  Mean MAE:       {cv.mean_mae:.4f} (±{cv.std_mae:.4f})")
    print(f"  Mean Test R²:   {cv.mean_r_squared:.4f} (±{cv.std_r_squared:.4f})")
    print(f"  Mean Train R²:  {cv.mean_train_r_squared:.4f}")
    gap = cv.mean_train_r_squared - cv.mean_r_squared
    if gap > 0.1:
        print(f"  Warning: Train R² >> Test R² (gap {gap:.3f}) — possible overfitting")
    else:
        print(f"  Good generalisation (train-test R² gap: {gap:.3f})")
    print()

    print(f"  {'Fold':>5}  {'Train':>6}  {'Test':>5}  {'RMSE':>10}  {'MAE':>8}  {'R²':>8}")
    print(f"  {'─'*50}")
    for fold in cv.fold_results:
        print(f"  {fold.fold_index:>5}  {fold.train_size:>6}  {fold.test_size:>5}  "
              f"{fold.rmse:>10.4f}  {fold.mae:>8.4f}  {fold.r_squared:>8.4f}")
    print()

    # ── 2. Ridge Lambda Selection ─────────────────────────────────────────────
    print("━━━ 2. Ridge — Lambda Selection via CV ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {'Lambda':>10}  {'Mean RMSE':>12}  {'Mean R²':>10}  {'Std RMSE':>10}")
    print(f"  {'─'*48}")
    best_ridge = {"lambda": None, "rmse": float("inf")}
    for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
        cv_r = linreg_core.kfold_cv_ridge(y, x_vars, lambda_val=lam, n_folds=5, shuffle=True, seed=42)
        marker = " <--" if cv_r.mean_rmse < best_ridge["rmse"] else ""
        if cv_r.mean_rmse < best_ridge["rmse"]:
            best_ridge = {"lambda": lam, "rmse": cv_r.mean_rmse}
        print(f"  {lam:>10.2f}  {cv_r.mean_rmse:>12.4f}  "
              f"{cv_r.mean_r_squared:>10.4f}  {cv_r.std_rmse:>10.4f}{marker}")
    print(f"\n  Best lambda: {best_ridge['lambda']:.2f}  (RMSE: {best_ridge['rmse']:.4f})")
    print()

    # ── 3. Lasso Lambda Selection ─────────────────────────────────────────────
    print("━━━ 3. Lasso — Lambda Selection via CV ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {'Lambda':>10}  {'Mean RMSE':>12}  {'Mean R²':>10}  {'Std RMSE':>10}")
    print(f"  {'─'*48}")
    best_lasso = {"lambda": None, "rmse": float("inf")}
    for lam in [0.01, 0.1, 0.5, 1.0, 2.0]:
        cv_l = linreg_core.kfold_cv_lasso(y, x_vars, lambda_val=lam, n_folds=5, shuffle=True, seed=42)
        if cv_l.mean_rmse < best_lasso["rmse"]:
            best_lasso = {"lambda": lam, "rmse": cv_l.mean_rmse}
        marker = " <--" if cv_l.mean_rmse == best_lasso["rmse"] else ""
        print(f"  {lam:>10.2f}  {cv_l.mean_rmse:>12.4f}  "
              f"{cv_l.mean_r_squared:>10.4f}  {cv_l.std_rmse:>10.4f}{marker}")
    print(f"\n  Best lambda: {best_lasso['lambda']:.2f}  (RMSE: {best_lasso['rmse']:.4f})")
    print()

    # ── 4. Elastic Net Alpha Selection ────────────────────────────────────────
    print("━━━ 4. Elastic Net — Alpha Selection via CV (λ=0.1) ━━━━━━━━━━━━━━━━")
    print("  Alpha: 0 = Ridge, 1 = Lasso")
    print(f"  {'Alpha':>8}  {'Mean RMSE':>12}  {'Mean R²':>10}  {'Std RMSE':>10}")
    print(f"  {'─'*46}")
    best_enet = {"alpha": None, "rmse": float("inf")}
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cv_e = linreg_core.kfold_cv_elastic_net(
            y, x_vars, lambda_val=0.1, alpha=alpha, n_folds=5, shuffle=True, seed=42
        )
        if cv_e.mean_rmse < best_enet["rmse"]:
            best_enet = {"alpha": alpha, "rmse": cv_e.mean_rmse}
        marker = " <--" if cv_e.mean_rmse == best_enet["rmse"] else ""
        print(f"  {alpha:>8.2f}  {cv_e.mean_rmse:>12.4f}  "
              f"{cv_e.mean_r_squared:>10.4f}  {cv_e.std_rmse:>10.4f}{marker}")
    print(f"\n  Best alpha: {best_enet['alpha']:.2f}  (RMSE: {best_enet['rmse']:.4f})")
    print()

    # ── 5. Coefficient Stability ──────────────────────────────────────────────
    print("━━━ 5. Coefficient Stability Across Folds ━━━━━━━━━━━━━━━━━━━━━━━━━━")
    # Re-run OLS CV without shuffle to get fold_coefficients
    cv_stab = linreg_core.kfold_cv_ols(y, x_vars, names, n_folds=5, shuffle=True, seed=42)
    all_coefs = cv_stab.fold_coefficients  # list of lists

    print(f"  {'Variable':<12}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}  Stability")
    print(f"  {'─'*64}")
    for i, name in enumerate(names):
        vals = [c[i] for c in all_coefs]
        mean_v = sum(vals) / len(vals)
        std_v  = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / len(vals))
        cv_val = (std_v / abs(mean_v)) if abs(mean_v) > 1e-10 else float("inf")
        status = "Very Stable" if cv_val < 0.1 else "Stable" if cv_val < 0.2 else "Variable"
        print(f"  {name:<12}  {mean_v:>10.4f}  {std_v:>10.4f}  "
              f"{min(vals):>10.4f}  {max(vals):>10.4f}  {status}")
    print()

    # ── 6. Reproducibility ────────────────────────────────────────────────────
    print("━━━ 6. Reproducibility with Fixed Seed ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    cv_a = linreg_core.kfold_cv_ols(y, x_vars, names, n_folds=4, shuffle=True, seed=12345)
    cv_b = linreg_core.kfold_cv_ols(y, x_vars, names, n_folds=4, shuffle=True, seed=12345)
    print(f"  Run 1 — RMSE: {cv_a.mean_rmse:.6f}  R²: {cv_a.mean_r_squared:.6f}")
    print(f"  Run 2 — RMSE: {cv_b.mean_rmse:.6f}  R²: {cv_b.mean_r_squared:.6f}")
    diff_rmse = abs(cv_a.mean_rmse - cv_b.mean_rmse)
    diff_r2   = abs(cv_a.mean_r_squared - cv_b.mean_r_squared)
    print(f"  Diff  — RMSE: {diff_rmse:.2e}  R²: {diff_r2:.2e}")
    print(f"  Identical results: {diff_rmse < 1e-12}")


if __name__ == "__main__":
    main()
