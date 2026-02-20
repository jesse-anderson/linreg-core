"""Prediction intervals example.

Run with:
    pip install linreg-core
    python prediction_intervals.py

Demonstrates:
- OLS prediction intervals (exact, using leverage)
- Ridge / Lasso / Elastic Net prediction intervals (conservative approximation)
- How interval width varies with confidence level
- How intervals widen for out-of-sample (extrapolation) points
"""

import linreg_core


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                    PREDICTION INTERVALS                              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # Training data: house size (sqft/100) vs price ($k)
    sqft  = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]
    price = [150.0, 175.0, 210.0, 240.0, 265.0, 295.0, 330.0, 360.0, 400.0, 430.0]

    # New points: in-sample range + one extrapolation at sqft=35
    new_sqft_vals = [11.0, 15.0, 19.0, 25.0, 35.0]
    new_x = [new_sqft_vals]   # list of column lists, matching x_vars format

    print("Training: 10 houses, sqft (hundreds) -> price ($k)")
    print(f"Predicting at: {new_sqft_vals}")
    print("Note: sqft=35 is extrapolation (training range: 10–28)")
    print()

    # ── 1. OLS Prediction Intervals ───────────────────────────────────────────
    print("━━━ 1. OLS Prediction Intervals (95%, exact) ━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Formula: PI = ŷ ± t(α/2, df) × √(MSE × (1 + leverage))")
    print()

    ols_pi = linreg_core.ols_prediction_intervals(price, [sqft], new_x, alpha=0.05)

    print(f"  {'SqFt':>8}  {'Predicted':>10}  {'Lower 95%':>10}  "
          f"{'Upper 95%':>10}  {'Width':>10}  {'Leverage':>8}")
    print(f"  {'─'*64}")
    for i, sx in enumerate(new_sqft_vals):
        width  = ols_pi.upper_bound[i] - ols_pi.lower_bound[i]
        marker = " <-- extrapolation" if sx > 28.0 else ""
        print(f"  {sx:>8.1f}  {ols_pi.predicted[i]:>10.2f}  "
              f"{ols_pi.lower_bound[i]:>10.2f}  {ols_pi.upper_bound[i]:>10.2f}  "
              f"{width:>10.2f}  {ols_pi.leverage[i]:>8.4f}{marker}")
    print()
    print("  Note: Interval width grows at the extrapolation point (sqft=35)")
    print("        because leverage is high far from the training data centre.")
    print(f"  df_residuals: {ols_pi.df_residuals:.1f}")
    print()

    # ── 2. Interval Width vs Confidence Level ─────────────────────────────────
    print("━━━ 2. Interval Width vs Confidence Level (at sqft=19) ━━━━━━━━━━━━━━")
    print(f"  {'Confidence':>12}  {'Lower':>10}  {'Predicted':>10}  "
          f"{'Upper':>10}  {'Width':>10}")
    print(f"  {'─'*56}")
    for conf in [0.50, 0.80, 0.90, 0.95, 0.99]:
        pi = linreg_core.ols_prediction_intervals(
            price, [sqft], [[19.0]], alpha=1.0 - conf
        )
        width = pi.upper_bound[0] - pi.lower_bound[0]
        print(f"  {conf*100:>11.0f}%  {pi.lower_bound[0]:>10.2f}  "
              f"{pi.predicted[0]:>10.2f}  {pi.upper_bound[0]:>10.2f}  {width:>10.2f}")
    print()

    # ── 3. Regularized Model Prediction Intervals ─────────────────────────────
    print("━━━ 3. Regularized Model Prediction Intervals (at sqft=19) ━━━━━━━━━━")
    print("  (Conservative approximation: unpenalized leverage + penalized MSE)")
    print()

    new_x_single = [[19.0]]

    ridge_pi = linreg_core.ridge_prediction_intervals(
        price, [sqft], new_x_single, alpha=0.05, lambda_val=1.0, standardize=True
    )
    lasso_pi = linreg_core.lasso_prediction_intervals(
        price, [sqft], new_x_single, alpha=0.05, lambda_val=0.5, standardize=True
    )
    enet_pi = linreg_core.elastic_net_prediction_intervals(
        price, [sqft], new_x_single, alpha=0.05,
        lambda_val=0.5, enet_alpha=0.5, standardize=True
    )

    print(f"  {'Method':<14}  {'Predicted':>10}  {'Lower 95%':>10}  "
          f"{'Upper 95%':>10}  {'Width':>10}")
    print(f"  {'─'*58}")

    # OLS at sqft=19 (index 2 in the original new_x prediction)
    ols_width = ols_pi.upper_bound[2] - ols_pi.lower_bound[2]
    print(f"  {'OLS (exact)':<14}  {ols_pi.predicted[2]:>10.2f}  "
          f"{ols_pi.lower_bound[2]:>10.2f}  {ols_pi.upper_bound[2]:>10.2f}  {ols_width:>10.2f}")

    r_width = ridge_pi.upper_bound[0] - ridge_pi.lower_bound[0]
    print(f"  {'Ridge':<14}  {ridge_pi.predicted[0]:>10.2f}  "
          f"{ridge_pi.lower_bound[0]:>10.2f}  {ridge_pi.upper_bound[0]:>10.2f}  {r_width:>10.2f}")

    l_width = lasso_pi.upper_bound[0] - lasso_pi.lower_bound[0]
    print(f"  {'Lasso':<14}  {lasso_pi.predicted[0]:>10.2f}  "
          f"{lasso_pi.lower_bound[0]:>10.2f}  {lasso_pi.upper_bound[0]:>10.2f}  {l_width:>10.2f}")

    e_width = enet_pi.upper_bound[0] - enet_pi.lower_bound[0]
    print(f"  {'Elastic Net':<14}  {enet_pi.predicted[0]:>10.2f}  "
          f"{enet_pi.lower_bound[0]:>10.2f}  {enet_pi.upper_bound[0]:>10.2f}  {e_width:>10.2f}")

    print()
    print("  Note: Regularized intervals are conservative (wider) because they")
    print("        use unpenalized leverage with the penalized model's MSE.")
    print()

    # ── 4. Standard Errors ────────────────────────────────────────────────────
    print("━━━ 4. OLS Prediction Standard Errors ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {'SqFt':>8}  {'Predicted':>10}  {'SE(pred)':>10}  {'Leverage':>10}")
    print(f"  {'─'*44}")
    for i, sx in enumerate(new_sqft_vals):
        print(f"  {sx:>8.1f}  {ols_pi.predicted[i]:>10.2f}  "
              f"{ols_pi.se_pred[i]:>10.4f}  {ols_pi.leverage[i]:>10.4f}")


if __name__ == "__main__":
    main()
