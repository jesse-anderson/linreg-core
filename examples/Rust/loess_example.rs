//! LOESS (Locally Estimated Scatterplot Smoothing) example.
//!
//! Demonstrates non-parametric regression using local polynomial fitting.
//! LOESS is useful for:
//! - Exploring non-linear relationships without assuming a functional form
//! - Smoothing noisy data to reveal underlying trends
//! - Fitting curves that adapt to local patterns in the data

use linreg_core::loess::{loess_fit, LoessOptions};
use linreg_core::loess::types::LoessSurface;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LOESS (Locally Estimated Scatterplot Smoothing) ===\n");

    // -------------------------------------------------------------------------
    // Example 1: Simple Linear Relationship
    // -------------------------------------------------------------------------
    println!("--- Example 1: Linear Relationship (y = 2x + 1) ---");
    let x1 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y1: Vec<f64> = x1.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    let options = LoessOptions {
        span: 0.75,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 1,
        surface: LoessSurface::Direct,
    };

    let fit1 = loess_fit(&y1, &[x1.clone()], &options)?;
    println!("Fitted values (first 5):");
    for i in 0..5 {
        println!("  x={:.1}, y={:.2}, fitted={:.2}", x1[i], y1[i], fit1.fitted[i]);
    }
    println!();

    // -------------------------------------------------------------------------
    // Example 2: Non-Linear Sine Wave
    // -------------------------------------------------------------------------
    println!("--- Example 2: Non-Linear Sine Wave ---");
    let x2: Vec<f64> = (0..=50).map(|i| i as f64 / 5.0).collect();
    let y2: Vec<f64> = x2.iter().map(|&xi| (xi * 0.5).sin() * 10.0 + 5.0).collect();

    // Compare different spans
    let spans = vec![0.3, 0.5, 0.75];
    println!("Comparing spans: smaller = wiggly, larger = smooth\n");

    for &span in &spans {
        let options_span = LoessOptions {
            span,
            degree: 1,
            robust_iterations: 0,
            n_predictors: 1,
            surface: LoessSurface::Direct,
        };
        let fit = loess_fit(&y2, &[x2.clone()], &options_span)?;

        // Compute residuals to show fit quality
        let sse: f64 = y2
            .iter()
            .zip(fit.fitted.iter())
            .map(|(y, y_hat)| (y - y_hat).powi(2))
            .sum();

        println!("Span={:.1}: SSE={:.2}", span, sse);
        println!("  Sample at x=2.0: y={:.2}, fitted={:.2}",
            x2.iter().zip(y2.iter()).find(|(x, _)| **x == 2.0).map(|(_, y)| *y).unwrap(),
            fit.fitted[x2.iter().position(|&x| (x - 2.0).abs() < 0.1).unwrap()]
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // Example 3: Linear vs Quadratic Degree
    // -------------------------------------------------------------------------
    println!("--- Example 3: Linear vs Quadratic Degree ---");
    let x3: Vec<f64> = (0..=20).map(|i| i as f64 / 2.0).collect();
    let y3: Vec<f64> = x3.iter().map(|&xi| 0.1 * xi * xi - xi + 5.0).collect();

    // Linear degree
    let options_linear = LoessOptions {
        span: 0.5,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 1,
        surface: LoessSurface::Direct,
    };
    let fit_linear = loess_fit(&y3, &[x3.clone()], &options_linear)?;

    // Quadratic degree
    let options_quad = LoessOptions {
        span: 0.5,
        degree: 2,
        robust_iterations: 0,
        n_predictors: 1,
        surface: LoessSurface::Direct,
    };
    let fit_quad = loess_fit(&y3, &[x3.clone()], &options_quad)?;

    // Compare at x=5.0 (where quadratic term matters)
    let idx_5 = x3.iter().position(|&x| (x - 5.0).abs() < 0.1).unwrap();
    println!("At x=5.0 (true y={:.2}):", y3[idx_5]);
    println!("  Linear degree fitted:  {:.2}", fit_linear.fitted[idx_5]);
    println!("  Quadratic degree fitted: {:.2}", fit_quad.fitted[idx_5]);
    println!();

    // -------------------------------------------------------------------------
    // Example 4: Prediction at New Points
    // -------------------------------------------------------------------------
    println!("--- Example 4: Prediction at New Points ---");
    let train_x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let train_y = vec![1.5, 2.1, 3.8, 5.2, 6.7, 8.1, 9.8, 11.2, 12.5, 14.1];

    let options_pred = LoessOptions {
        span: 0.6,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 1,
        surface: LoessSurface::Direct,
    };
    let fit_pred = loess_fit(&train_y, &[train_x.clone()], &options_pred)?;

    // Predict at new points
    let new_x = vec![1.5, 3.5, 5.5, 7.5];
    let predictions = fit_pred.predict(&[new_x.clone()], &[train_x], &train_y, &options_pred)?;

    println!("Predictions at new points:");
    for (i, &x) in new_x.iter().enumerate() {
        println!("  x={:.1}: predicted={:.2}", x, predictions[i]);
    }
    println!();

    // -------------------------------------------------------------------------
    // Example 5: Multiple Predictors (Bivariate LOESS)
    // -------------------------------------------------------------------------
    println!("--- Example 5: Multiple Predictors ---");
    let x1_5 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let x2_5 = vec![5.0, 3.0, 8.0, 2.0, 7.0, 1.0, 6.0, 4.0, 9.0, 0.0];
    // y is sum of predictors plus noise
    let y5: Vec<f64> = x1_5
        .iter()
        .zip(x2_5.iter())
        .map(|(&a, &b)| a + b + 0.5)
        .collect();

    let options_multi = LoessOptions {
        span: 0.7,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 2,
        surface: LoessSurface::Direct,
    };
    let fit_multi = loess_fit(&y5, &[x1_5.clone(), x2_5.clone()], &options_multi)?;

    println!("Bivariate LOESS (2 predictors):");
    println!("  Sample observations:");
    for i in 0..5 {
        println!("    x1={:.1}, x2={:.1}, y={:.2}, fitted={:.2}",
            x1_5[i], x2_5[i], y5[i], fit_multi.fitted[i]);
    }
    println!();

    // -------------------------------------------------------------------------
    // Example 6: Effect of Span on Smoothness
    // -------------------------------------------------------------------------
    println!("--- Example 6: Span Parameter Effect ---");
    let x6: Vec<f64> = (0..=30).map(|i| i as f64).collect();
    // Noisy data with trend
    let y6: Vec<f64> = x6
        .iter()
        .map(|&xi| 0.5 * xi + 5.0 + (xi * 0.3).cos() * 3.0)
        .collect();

    println!("Fitting with different span values:");
    for &span in &[0.2, 0.4, 0.6, 0.8] {
        let opt = LoessOptions {
            span,
            degree: 1,
            robust_iterations: 0,
            n_predictors: 1,
            surface: LoessSurface::Direct,
        };
        let fit = loess_fit(&y6, &[x6.clone()], &opt)?;
        let mse: f64 = y6
            .iter()
            .zip(fit.fitted.iter())
            .map(|(y, y_hat)| (y - y_hat).powi(2))
            .sum::<f64>()
            / y6.len() as f64;
        println!("  Span={:.1}: MSE={:.3}", span, mse);
    }
    println!();

    // -------------------------------------------------------------------------
    // Key Takeaways
    // -------------------------------------------------------------------------
    println!("=== Key LOESS Parameters ===");
    println!("span:     Fraction of data used for each local fit");
    println!("          - Smaller (0.2-0.3): Wiggly, follows data closely");
    println!("          - Medium (0.5-0.6): Balanced smoothness");
    println!("          - Larger (0.7-0.9): Very smooth, may underfit");
    println!();
    println!("degree:   Polynomial degree for local fits");
    println!("          - 1: Linear (faster, less flexible)");
    println!("          - 2: Quadratic (slower, more flexible)");
    println!();
    println!("When to use LOESS:");
    println!("  - Exploring unknown relationships");
    println!("  - Data visualization and smoothing");
    println!("  - When parametric models don't fit well");
    println!("  - With small to medium datasets (< 10,000 points)");

    Ok(())
}
