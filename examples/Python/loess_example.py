"""LOESS (Locally Estimated Scatterplot Smoothing) example.

Run with:
    pip install linreg-core
    python loess_example.py

Demonstrates:
- Fitting LOESS to linear and non-linear data
- Effect of span parameter on smoothness
- Linear (degree=1) vs quadratic (degree=2) local fits
- Prediction at new points using loess_predict
"""

import math
import linreg_core


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║         LOESS REGRESSION — PYTHON BINDINGS                         ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # ── 1. Simple Linear Relationship ─────────────────────────────────────────
    print("━━━ 1. Linear Relationship (y = 2x + 1) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    x1 = [float(i) for i in range(101)]          # 0 .. 100
    y1 = [2.0 * xi + 1.0 for xi in x1]

    fit1 = linreg_core.loess_fit(y1, [x1], span=0.75, degree=1)
    print(f"  {'x':>5}  {'y (true)':>10}  {'fitted':>10}  {'residual':>10}")
    print(f"  {'─'*40}")
    for i in range(5):
        print(f"  {x1[i]:>5.1f}  {y1[i]:>10.2f}  {fit1.fitted[i]:>10.2f}  "
              f"{fit1.residuals[i]:>10.4f}")
    print(f"  MSE:  {fit1.mse:.6f}  (should be ~0 for perfect linear data)")
    print()

    # ── 2. Non-Linear Sine Wave — Comparing Spans ─────────────────────────────
    print("━━━ 2. Non-Linear Sine Wave — Effect of Span ━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Smaller span = wigglier fit, larger span = smoother fit")
    print()

    x2 = [i / 5.0 for i in range(501)]           # 0 .. 100 in steps of 0.2
    y2 = [math.sin(xi * 0.5) * 10.0 + 5.0 for xi in x2]

    print(f"  {'Span':>6}  {'MSE':>10}  {'RMSE':>10}")
    print(f"  {'─'*30}")
    for span in [0.1, 0.3, 0.5, 0.75]:
        fit = linreg_core.loess_fit(y2, [x2], span=span, degree=1)
        print(f"  {span:>6.2f}  {fit.mse:>10.4f}  {fit.rmse:>10.4f}")
    print()

    # ── 3. Degree 1 vs Degree 2 on a Quadratic Curve ──────────────────────────
    print("━━━ 3. Linear vs Quadratic Degree on y = 0.1x² − x + 5 ━━━━━━━━━━━━")
    x3 = [i / 2.0 for i in range(201)]           # 0 .. 100 in steps of 0.5
    y3 = [0.1 * xi**2 - xi + 5.0 for xi in x3]

    fit_lin  = linreg_core.loess_fit(y3, [x3], span=0.5, degree=1)
    fit_quad = linreg_core.loess_fit(y3, [x3], span=0.5, degree=2)

    # Find index closest to x=5.0
    idx5 = min(range(len(x3)), key=lambda i: abs(x3[i] - 5.0))
    true_y5 = y3[idx5]
    print(f"  At x=5.0 (true y = {true_y5:.2f}):")
    print(f"    Degree 1 fitted:  {fit_lin.fitted[idx5]:.4f}  "
          f"(error {abs(fit_lin.fitted[idx5] - true_y5):.4f})")
    print(f"    Degree 2 fitted:  {fit_quad.fitted[idx5]:.4f}  "
          f"(error {abs(fit_quad.fitted[idx5] - true_y5):.4f})")
    print()
    print(f"  Overall MSE — degree 1: {fit_lin.mse:.4f}   "
          f"degree 2: {fit_quad.mse:.4f}")
    print()

    # ── 4. Prediction at New Points ───────────────────────────────────────────
    print("━━━ 4. Prediction at New Points ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    train_x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    train_y = [1.5, 2.1, 3.8, 5.2, 6.7, 8.1, 9.8, 11.2, 12.5, 14.1]
    new_x   = [1.5, 3.5, 5.5, 7.5]

    preds = linreg_core.loess_predict(
        new_x, train_x, train_y, span=0.6, degree=1
    )
    print(f"  {'x':>6}  {'predicted':>12}")
    print(f"  {'─'*22}")
    for xi, yi in zip(new_x, preds):
        print(f"  {xi:>6.1f}  {yi:>12.4f}")
    print()

    # ── 5. Span Sweep — MSE Table ─────────────────────────────────────────────
    print("━━━ 5. Span Sweep on Noisy Trend (span 0.2 -> 0.8) ━━━━━━━━━━━━━━━━━")
    x5 = [float(i) for i in range(301)]          # 0 .. 300
    y5 = [0.5 * xi + 5.0 + math.cos(xi * 0.3) * 3.0 for xi in x5]

    print(f"  {'Span':>6}  {'MSE':>10}  {'RMSE':>10}  Notes")
    print(f"  {'─'*50}")
    for span in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        fit = linreg_core.loess_fit(y5, [x5], span=span, degree=1)
        note = ""
        if span <= 0.3:
            note = "← wiggly"
        elif span >= 0.7:
            note = "← very smooth"
        print(f"  {span:>6.1f}  {fit.mse:>10.4f}  {fit.rmse:>10.4f}  {note}")
    print()

    # ── Key Parameters Reference ──────────────────────────────────────────────
    print("━━━ Key LOESS Parameters ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  span    Fraction of data used per local fit  (0 < span ≤ 1)")
    print("            0.2-0.3  Wiggly, follows data closely")
    print("            0.5-0.6  Balanced smoothness")
    print("            0.7-0.9  Very smooth, may underfit")
    print()
    print("  degree  Polynomial degree for local fits")
    print("            1  Linear (faster, adequate for mild curvature)")
    print("            2  Quadratic (handles stronger curvature)")
    print()
    print("  When to use LOESS:")
    print("    - Exploring unknown or non-linear relationships")
    print("    - Smoothing noisy data to reveal underlying trend")
    print("    - Small to medium datasets (n < 10 000)")


if __name__ == "__main__":
    main()
