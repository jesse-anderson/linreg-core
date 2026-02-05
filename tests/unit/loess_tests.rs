// ============================================================================
// LOESS Tests
// ============================================================================
//
// Comprehensive unit tests for LOESS (Locally Estimated Scatterplot Smoothing)

use linreg_core::loess::{loess_fit, LoessOptions};

/// Helper: Generate simple linear data (y = 2x + 1)
fn simple_linear_data() -> (Vec<f64>, Vec<Vec<f64>>) {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi + 1.0).collect();
    (y, vec![x])
}

/// Helper: Generate sinusoid data for testing non-linear smoothing
fn sinusoid_data() -> (Vec<f64>, Vec<Vec<f64>>) {
    let x: Vec<f64> = (0..=100).map(|i| i as f64 * 0.1).collect();
    let y: Vec<f64> = x.iter().map(|xi| (xi).sin()).collect();
    (y, vec![x])
}

/// Helper: Generate quadratic data (y = x² - 3x + 2)
fn quadratic_data() -> (Vec<f64>, Vec<Vec<f64>>) {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y: Vec<f64> = x.iter().map(|xi| xi * xi - 3.0 * xi + 2.0).collect();
    (y, vec![x])
}

/// Helper: Generate data with multiple predictors
fn multiple_predictor_data() -> (Vec<f64>, Vec<Vec<f64>>) {
    let x1 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let x2 = vec![5.0, 3.0, 8.0, 2.0, 7.0, 1.0, 6.0, 4.0, 9.0, 0.0];
    let y: Vec<f64> = x1.iter().zip(x2.iter()).map(|(&a, &b)| a + b).collect();
    (y, vec![x1, x2])
}

/// Helper: Generate small dataset for edge case testing
fn small_data() -> (Vec<f64>, Vec<Vec<f64>>) {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    (y, vec![x])
}

#[test]
fn test_loess_basic() {
    let (y, x) = simple_linear_data();
    let options = LoessOptions::default();

    let result = loess_fit(&y, &x, &options);

    assert!(result.is_ok());
    let fit = result.unwrap();
    assert_eq!(fit.fitted.len(), y.len());
    assert_eq!(fit.span, 0.75);
    assert_eq!(fit.degree, 1);
}

#[test]
fn test_loess_different_spans() {
    let (y, x) = sinusoid_data();

    // Small span = wiggly (more local)
    let options_small = LoessOptions {
        span: 0.25,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 1,
        surface: linreg_core::loess::types::LoessSurface::Direct,
    };
    let result_small = loess_fit(&y, &x, &options_small).unwrap();

    // Medium span
    let options_medium = LoessOptions {
        span: 0.5,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 1,
        surface: linreg_core::loess::types::LoessSurface::Direct,
    };
    let result_medium = loess_fit(&y, &x, &options_medium).unwrap();

    // Large span = smooth (more global)
    let options_large = LoessOptions {
        span: 0.75,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 1,
        surface: linreg_core::loess::types::LoessSurface::Direct,
    };
    let result_large = loess_fit(&y, &x, &options_large).unwrap();

    // Full span = very smooth
    let options_full = LoessOptions {
        span: 1.0,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 1,
        surface: linreg_core::loess::types::LoessSurface::Direct,
    };
    let result_full = loess_fit(&y, &x, &options_full).unwrap();

    // All should produce valid fits
    assert_eq!(result_small.fitted.len(), y.len());
    assert_eq!(result_medium.fitted.len(), y.len());
    assert_eq!(result_large.fitted.len(), y.len());
    assert_eq!(result_full.fitted.len(), y.len());

    // Verify span values are stored correctly
    assert_eq!(result_small.span, 0.25);
    assert_eq!(result_medium.span, 0.5);
    assert_eq!(result_large.span, 0.75);
    assert_eq!(result_full.span, 1.0);
}

#[test]
fn test_loess_multiple_predictors() {
    let (y, x) = multiple_predictor_data();

    let options = LoessOptions {
        span: 0.75,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 2,
        surface: linreg_core::loess::types::LoessSurface::Direct,
    };

    let result = loess_fit(&y, &x, &options);

    assert!(result.is_ok());
    let fit = result.unwrap();
    assert_eq!(fit.fitted.len(), y.len());

    // All fitted values should be finite
    for &val in &fit.fitted {
        assert!(val.is_finite());
    }
}

#[test]
fn test_loess_quadratic_degree() {
    let (y, x) = quadratic_data();

    // Linear fit
    let options_linear = LoessOptions {
        span: 0.75,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 1,
        surface: linreg_core::loess::types::LoessSurface::Direct,
    };
    let result_linear = loess_fit(&y, &x, &options_linear).unwrap();

    // Quadratic fit
    let options_quadratic = LoessOptions {
        span: 0.75,
        degree: 2,
        robust_iterations: 0,
        n_predictors: 1,
        surface: linreg_core::loess::types::LoessSurface::Direct,
    };
    let result_quadratic = loess_fit(&y, &x, &options_quadratic).unwrap();

    // Both should produce valid fits
    assert_eq!(result_linear.degree, 1);
    assert_eq!(result_quadratic.degree, 2);

    // Quadratic should fit the parabola better than linear
    // Compare mean absolute error for interior points
    let mae_linear: f64 = result_linear.fitted[3..8]
        .iter()
        .zip(&y[3..8])
        .map(|(&f, &t)| (f - t).abs())
        .sum::<f64>()
        / 5.0;

    let mae_quadratic: f64 = result_quadratic.fitted[3..8]
        .iter()
        .zip(&y[3..8])
        .map(|(&f, &t)| (f - t).abs())
        .sum::<f64>()
        / 5.0;

    // Quadratic should have lower error for this quadratic data
    assert!(mae_quadratic < mae_linear);
}

#[test]
fn test_loess_edge_cases() {
    // Test with small data
    let (y, x) = small_data();
    let options = LoessOptions::default();

    let result = loess_fit(&y, &x, &options);
    if let Err(e) = &result {
        eprintln!("Error fitting LOESS with small data: {:?}", e);
    }
    assert!(result.is_ok(), "LOESS fit with small data should succeed");
    let fit = result.unwrap();
    assert_eq!(fit.fitted.len(), y.len());
}

#[test]
fn test_loess_insufficient_data() {
    // Test with only 1 point (n=1) - should fail
    let x = vec![0.0];
    let y = vec![0.0];

    let options = LoessOptions::default();
    let result = loess_fit(&y, &[x], &options);

    assert!(result.is_err());
}

#[test]
fn test_loess_invalid_span() {
    let (y, x) = simple_linear_data();

    // Span > 1.0 is invalid
    let options_high = LoessOptions {
        span: 1.5,
        ..Default::default()
    };
    assert!(loess_fit(&y, &x, &options_high).is_err());

    // Span = 0.0 is invalid
    let options_zero = LoessOptions {
        span: 0.0,
        ..Default::default()
    };
    assert!(loess_fit(&y, &x, &options_zero).is_err());

    // Negative span is invalid
    let options_neg = LoessOptions {
        span: -0.1,
        ..Default::default()
    };
    assert!(loess_fit(&y, &x, &options_neg).is_err());
}

#[test]
fn test_loess_invalid_degree() {
    let (y, x) = simple_linear_data();

    // Degree > 2 is invalid
    let options_3 = LoessOptions {
        degree: 3,
        ..Default::default()
    };
    assert!(loess_fit(&y, &x, &options_3).is_err());

    // Degree 0 is now valid (constant model)
    let options_0 = LoessOptions {
        degree: 0,
        ..Default::default()
    };
    assert!(loess_fit(&y, &x, &options_0).is_ok());
}

#[test]
fn test_loess_dimension_mismatch() {
    let x1 = vec![0.0, 1.0, 2.0, 3.0];
    let x2 = vec![0.0, 1.0, 2.0]; // Wrong length!
    let y = vec![0.0, 1.0, 2.0, 3.0];

    let options = LoessOptions::default();
    let result = loess_fit(&y, &[x1, x2], &options);

    assert!(result.is_err());
}

#[test]
fn test_loess_empty_predictors() {
    let y = vec![1.0, 2.0, 3.0];

    let options = LoessOptions::default();
    let result = loess_fit(&y, &[], &options);

    assert!(result.is_err());
}

#[test]
fn test_loess_prediction() {
    let (train_y, train_x) = simple_linear_data();
    let options = LoessOptions::default();
    let fit = loess_fit(&train_y, &train_x, &options).unwrap();

    // Predict at new points
    let new_x = vec![1.5, 3.5, 5.5, 7.5];
    let predictions = fit.predict(&[new_x], &train_x, &train_y, &options).unwrap();

    assert_eq!(predictions.len(), 4);

    // Predictions should be close to true values: y = 2*x + 1
    // Expected: [4.0, 8.0, 12.0, 16.0]
    assert!((predictions[0] - 4.0).abs() < 1.0);
    assert!((predictions[1] - 8.0).abs() < 1.0);
    assert!((predictions[2] - 12.0).abs() < 1.0);
    assert!((predictions[3] - 16.0).abs() < 1.0);
}

#[test]
fn test_loess_prediction_span_mismatch() {
    let (train_y, train_x) = simple_linear_data();

    let fit_options = LoessOptions {
        span: 0.75,
        ..Default::default()
    };
    let fit = loess_fit(&train_y, &train_x, &fit_options).unwrap();

    // Try to predict with different span
    let predict_options = LoessOptions {
        span: 0.5,
        ..Default::default()
    };

    let new_x = vec![2.5];
    let result = fit.predict(&[new_x], &train_x, &train_y, &predict_options);

    assert!(result.is_err());
}

#[test]
fn test_loess_prediction_degree_mismatch() {
    let (train_y, train_x) = simple_linear_data();

    let fit_options = LoessOptions {
        degree: 1,
        ..Default::default()
    };
    let fit = loess_fit(&train_y, &train_x, &fit_options).unwrap();

    // Try to predict with different degree
    let predict_options = LoessOptions {
        degree: 2,
        ..Default::default()
    };

    let new_x = vec![2.5];
    let result = fit.predict(&[new_x], &train_x, &train_y, &predict_options);

    assert!(result.is_err());
}

#[test]
fn test_loess_prediction_multiple_predictors() {
    let (train_y, train_x) = multiple_predictor_data();

    let options = LoessOptions {
        span: 0.75,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 2,
        surface: linreg_core::loess::types::LoessSurface::Direct,
    };
    let fit = loess_fit(&train_y, &train_x, &options).unwrap();

    // Predict at new points
    let new_x1 = vec![2.5, 5.5];
    let new_x2 = vec![4.0, 3.0];
    let predictions = fit
        .predict(&[new_x1, new_x2], &train_x, &train_y, &options)
        .unwrap();

    assert_eq!(predictions.len(), 2);
    assert!(predictions[0].is_finite());
    assert!(predictions[1].is_finite());
}

#[test]
fn test_loess_prediction_empty() {
    let (train_y, train_x) = simple_linear_data();
    let options = LoessOptions::default();
    let fit = loess_fit(&train_y, &train_x, &options).unwrap();

    // Empty prediction should return empty vector
    let new_x: Vec<f64> = vec![];
    let predictions = fit.predict(&[new_x], &train_x, &train_y, &options).unwrap();

    assert!(predictions.is_empty());
}

#[test]
fn test_loess_extrapolation() {
    let train_x = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let train_y: Vec<f64> = train_x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    let options = LoessOptions {
        span: 0.75,
        ..Default::default()
    };
    let fit = loess_fit(&train_y, &[train_x.clone()], &options).unwrap();

    // Predict outside training range
    let new_x = vec![0.5, 9.5]; // Below and above training range
    let predictions = fit.predict(&[new_x], &[train_x], &train_y, &options).unwrap();

    assert_eq!(predictions.len(), 2);
    // Extrapolation should still produce finite values
    assert!(predictions[0].is_finite());
    assert!(predictions[1].is_finite());
}

#[test]
fn test_loess_constant_y() {
    // Test with constant response variable
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let y = vec![5.0; 10];

    let options = LoessOptions::default();
    let result = loess_fit(&y, &[x.clone()], &options);

    assert!(result.is_ok());
    let fit = result.unwrap();
    assert_eq!(fit.fitted.len(), 10);

    // All fitted values should be close to 5.0
    for &val in &fit.fitted {
        assert!((val - 5.0).abs() < 1.0);
    }
}

#[test]
fn test_loess_monotonic_data() {
    // Test with strictly monotonic data
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let y: Vec<f64> = x.iter().map(|&xi| xi * xi * xi).collect();

    let options = LoessOptions {
        span: 0.75,
        ..Default::default()
    };
    let result = loess_fit(&y, &[x], &options);

    assert!(result.is_ok());
    let fit = result.unwrap();

    // Fitted values should generally follow the trend
    // (interior points should be reasonably close)
    for i in 3..7 {
        assert!((fit.fitted[i] - y[i]).abs() < 50.0);
    }
}

#[test]
fn test_loess_with_noise() {
    // Test with noisy data
    let x: Vec<f64> = (0..=50).map(|i| i as f64 * 0.2).collect();
    // Use deterministic "noise" instead of random for reproducibility
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| xi.sin() + ((i as f64 * 0.1).sin() * 0.1))
        .collect();

    let options = LoessOptions::default();
    let result = loess_fit(&y, &[x.clone()], &options);

    assert!(result.is_ok());
    let fit = result.unwrap();

    // Fitted values should smooth the noise
    // Check that fitted values are within reasonable bounds
    for &val in &fit.fitted {
        assert!(val > -2.0 && val < 2.0);
    }
}

#[test]
fn test_loess_options_clone() {
    let options = LoessOptions {
        span: 0.5,
        degree: 2,
        robust_iterations: 0,
        n_predictors: 3,
        surface: linreg_core::loess::types::LoessSurface::Direct,
    };

    let cloned = options.clone();
    assert_eq!(options.span, cloned.span);
    assert_eq!(options.degree, cloned.degree);
    assert_eq!(options.n_predictors, cloned.n_predictors);
}

#[test]
fn test_loess_fit_clone() {
    let (y, x) = simple_linear_data();
    let options = LoessOptions::default();
    let fit = loess_fit(&y, &x, &options).unwrap();

    let cloned = fit.clone();
    assert_eq!(fit.fitted.len(), cloned.fitted.len());
    assert_eq!(fit.span, cloned.span);
    assert_eq!(fit.degree, cloned.degree);
}

#[test]
fn test_loess_three_predictors() {
    // Test with 3 predictors - ensure they're not collinear
    let x1 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let x2 = vec![5.0, 3.0, 8.0, 2.0, 7.0, 1.0, 6.0, 4.0, 9.0, 0.0, 5.5];
    let x3 = vec![2.0, 7.0, 1.0, 8.0, 0.5, 9.0, 3.0, 6.0, 1.5, 8.5, 4.0];
    let y: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .zip(x3.iter())
        .map(|((&a, &b), &c)| a + 0.5 * b - 0.3 * c)
        .collect();

    let options = LoessOptions {
        span: 0.75,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 3,
        surface: linreg_core::loess::types::LoessSurface::Direct,
    };

    let result = loess_fit(&y, &[x1, x2, x3], &options);

    if let Err(e) = &result {
        eprintln!("Error fitting LOESS with 3 predictors: {:?}", e);
    }
    assert!(result.is_ok(), "LOESS fit with 3 predictors should succeed");
    let fit = result.unwrap();
    assert_eq!(fit.fitted.len(), y.len());

    // All fitted values should be finite
    for &val in &fit.fitted {
        assert!(val.is_finite());
    }
}
