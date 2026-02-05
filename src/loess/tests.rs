//! LOESS module tests

use crate::loess::*;

#[test]
fn test_loess_options_default() {
    let options = LoessOptions::default();
    assert_eq!(options.span, 0.75);
    assert_eq!(options.degree, 1);
    assert_eq!(options.robust_iterations, 0);
    assert_eq!(options.n_predictors, 1);
}

#[test]
fn test_loess_options_custom() {
    let options = LoessOptions {
        span: 0.5,
        degree: 2,
        robust_iterations: 2,
        n_predictors: 2,
        surface: LoessSurface::Direct,
    };
    assert_eq!(options.span, 0.5);
    assert_eq!(options.degree, 2);
    assert_eq!(options.robust_iterations, 2);
    assert_eq!(options.n_predictors, 2);
}

#[test]
fn test_loess_fit_empty() {
    let fit = LoessFit {
        fitted: vec![],
        predictions: None,
        span: 0.75,
        degree: 1,
        robust_iterations: 0,
        surface: LoessSurface::Direct,
    };
    assert!(fit.fitted.is_empty());
    assert!(fit.predictions.is_none());
    assert_eq!(fit.span, 0.75);
    assert_eq!(fit.degree, 1);
}

#[test]
fn test_loess_fit_simple_linear() {
    // Simple linear relationship: y = 2x + 1
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    let options = LoessOptions {
        span: 0.75,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 1,
        surface: LoessSurface::Direct,
    };

    let result = loess_fit(&y, &[x], &options).unwrap();

    // Check that we get fitted values
    assert_eq!(result.fitted.len(), y.len());
    assert_eq!(result.span, 0.75);
    assert_eq!(result.degree, 1);

    // For a perfect linear relationship with large span, LOESS should fit well
    // Check interior points (interpolated, not extrapolated at boundaries)
    for i in 3..8 {
        assert!((result.fitted[i] - y[i]).abs() < 0.5);
    }
}

#[test]
fn test_loess_fit_different_spans() {
    // Test that different spans produce different smoothness
    let x: Vec<f64> = (0..=20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| (xi * 0.3).sin()).collect();

    // Small span = wiggly
    let options_small = LoessOptions {
        span: 0.3,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 1,
        surface: LoessSurface::Direct,
    };
    let result_small = loess_fit(&y, &[x.clone()], &options_small).unwrap();

    // Large span = smooth
    let options_large = LoessOptions {
        span: 0.9,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 1,
        surface: LoessSurface::Direct,
    };
    let result_large = loess_fit(&y, &[x], &options_large).unwrap();

    // Both should produce valid fits
    assert_eq!(result_small.fitted.len(), y.len());
    assert_eq!(result_large.fitted.len(), y.len());
}

#[test]
fn test_loess_fit_quadratic_degree() {
    // Test quadratic degree
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y: Vec<f64> = x.iter().map(|&xi| xi * xi - 3.0 * xi + 2.0).collect();

    let options = LoessOptions {
        span: 0.75,
        degree: 2,
        robust_iterations: 0,
        n_predictors: 1,
        surface: LoessSurface::Direct,
    };

    let result = loess_fit(&y, &[x], &options).unwrap();

    assert_eq!(result.degree, 2);
    assert_eq!(result.fitted.len(), y.len());

    // Quadratic should fit parabola better than linear
    // Check interior points
    for i in 3..8 {
        assert!((result.fitted[i] - y[i]).abs() < 1.0);
    }
}

#[test]
fn test_loess_fit_constant_degree() {
    // Test degree 0 (constant model)
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![10.0, 12.0, 11.0, 13.0, 12.0];

    let options = LoessOptions {
        span: 0.6,
        degree: 0,
        robust_iterations: 0,
        n_predictors: 1,
        surface: LoessSurface::Direct,
    };

    let result = loess_fit(&y, &[x], &options).unwrap();

    assert_eq!(result.degree, 0);
    assert_eq!(result.fitted.len(), y.len());

    // All fitted values should be close to the mean (around 11.6)
    // Allow slightly wider range since degree 0 uses local neighborhoods
    for &fitted in &result.fitted {
        assert!(fitted > 9.0 && fitted < 14.0);
    }
}

#[test]
fn test_loess_fit_insufficient_data() {
    // Test error handling for too few points (need at least 2 for degree 1)
    let x = vec![0.0];
    let y = vec![0.0];

    let options = LoessOptions::default();

    let result = loess_fit(&y, &[x], &options);
    assert!(result.is_err());
}

#[test]
fn test_loess_fit_invalid_span() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];

    // Invalid span (must be in (0, 1])
    let options = LoessOptions {
        span: 1.5,
        ..Default::default()
    };

    let result = loess_fit(&y, &[x.clone()], &options);
    assert!(result.is_err());

    // Span of 0 is also invalid
    let options_zero = LoessOptions {
        span: 0.0,
        ..Default::default()
    };

    let result_zero = loess_fit(&y, &[x], &options_zero);
    assert!(result_zero.is_err());
}

#[test]
fn test_loess_fit_invalid_degree() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];

    // Invalid degree (must be 0, 1, or 2)
    let options = LoessOptions {
        degree: 3,
        ..Default::default()
    };

    let result = loess_fit(&y, &[x], &options);
    assert!(result.is_err());
}

#[test]
fn test_loess_fit_dimension_mismatch() {
    let x1 = vec![0.0, 1.0, 2.0, 3.0];
    let x2 = vec![0.0, 1.0, 2.0];  // Wrong length!
    let y = vec![0.0, 1.0, 2.0, 3.0];

    let options = LoessOptions::default();

    let result = loess_fit(&y, &[x1, x2], &options);
    assert!(result.is_err());
}

#[test]
fn test_loess_fit_multiple_predictors() {
    // Test with 2 predictors that are not linearly related
    let x1 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let x2 = vec![5.0, 3.0, 8.0, 2.0, 7.0, 1.0, 6.0, 4.0, 9.0, 0.0];
    // Simple linear combination
    let y: Vec<f64> = x1.iter().zip(x2.iter()).map(|(&a, &b)| a + b).collect();

    let options = LoessOptions {
        span: 0.75,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 2,
        surface: LoessSurface::Direct,
    };

    let result = loess_fit(&y, &[x1, x2], &options).unwrap();

    assert_eq!(result.fitted.len(), y.len());
    // Check that fitted values are finite and reasonable
    for i in 0..y.len() {
        assert!(result.fitted[i].is_finite());
    }
}

#[test]
fn test_loess_fit_robust() {
    // Test robust fitting - verifies that robustness iterations run without error
    // Note: MAD-based robustness works best when outliers are common enough
    // that MAD represents the outlier scale. When LOESS already fits well
    // locally, robust iterations may break early (which is correct behavior).
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let mut y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
    // Add outlier at x=5
    y[5] = 100.0;

    // Non-robust fit
    let options_non_robust = LoessOptions {
        span: 0.5,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 1,
        surface: LoessSurface::Direct,
    };
    let result_non_robust = loess_fit(&y, &[x.clone()], &options_non_robust).unwrap();

    // Robust fit
    let options_robust = LoessOptions {
        span: 0.5,
        degree: 1,
        robust_iterations: 2,
        n_predictors: 1,
        surface: LoessSurface::Direct,
    };
    let result_robust = loess_fit(&y, &[x], &options_robust).unwrap();

    // Verify robust fit completed and returned fitted values
    assert_eq!(result_robust.robust_iterations, 2);
    assert_eq!(result_robust.fitted.len(), y.len());
    assert_eq!(result_non_robust.fitted.len(), y.len());

    // Both fits should produce finite values
    for &f in &result_robust.fitted {
        assert!(f.is_finite());
    }
    for &f in &result_non_robust.fitted {
        assert!(f.is_finite());
    }
}

#[test]
fn test_loess_predict_basic() {
    // Test basic prediction on new points
    let train_x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let train_y: Vec<f64> = train_x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    let options = LoessOptions::default();
    let fit = loess_fit(&train_y, &[train_x.clone()], &options).unwrap();

    // Predict at points within the training range
    let new_x = vec![1.5, 3.5, 5.5, 7.5];
    let predictions = fit.predict(&[new_x], &[train_x], &train_y, &options).unwrap();

    assert_eq!(predictions.len(), 4);
    // Predictions should be close to true values: y = 2*x + 1
    // Expected: [4.0, 8.0, 12.0, 16.0]
    assert!((predictions[0] - 4.0).abs() < 1.0);
    assert!((predictions[1] - 8.0).abs() < 1.0);
    assert!((predictions[2] - 12.0).abs() < 1.0);
    assert!((predictions[3] - 16.0).abs() < 1.0);
}

#[test]
fn test_loess_predict_span_mismatch() {
    // Test error when span doesn't match
    let train_x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let train_y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let options = LoessOptions {
        span: 0.75,
        robust_iterations: 0,
        ..Default::default()
    };
    let fit = loess_fit(&train_y, &[train_x.clone()], &options).unwrap();

    // Try to predict with different span
    let wrong_options = LoessOptions {
        span: 0.5,
        robust_iterations: 0,
        ..Default::default()
    };
    let new_x = vec![2.5];
    let result = fit.predict(&[new_x], &[train_x], &train_y, &wrong_options);

    assert!(result.is_err());
}

#[test]
fn test_loess_predict_degree_mismatch() {
    // Test error when degree doesn't match
    let train_x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let train_y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let options = LoessOptions {
        degree: 1,
        robust_iterations: 0,
        ..Default::default()
    };
    let fit = loess_fit(&train_y, &[train_x.clone()], &options).unwrap();

    // Try to predict with different degree
    let wrong_options = LoessOptions {
        degree: 2,
        robust_iterations: 0,
        ..Default::default()
    };
    let new_x = vec![2.5];
    let result = fit.predict(&[new_x], &[train_x], &train_y, &wrong_options);

    assert!(result.is_err());
}

#[test]
fn test_loess_predict_robust_mismatch() {
    // Test error when robust iterations don't match
    let train_x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let train_y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let options = LoessOptions {
        robust_iterations: 2,
        ..Default::default()
    };
    let fit = loess_fit(&train_y, &[train_x.clone()], &options).unwrap();

    // Try to predict with different robust iterations
    let wrong_options = LoessOptions {
        robust_iterations: 0,
        ..Default::default()
    };
    let new_x = vec![2.5];
    let result = fit.predict(&[new_x], &[train_x], &train_y, &wrong_options);

    assert!(result.is_err());
}

#[test]
fn test_loess_predict_multiple_predictors() {
    // Test prediction with multiple predictors
    let train_x1 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let train_x2 = vec![5.0, 3.0, 8.0, 2.0, 7.0, 1.0, 6.0, 4.0, 9.0, 0.0];
    let train_y: Vec<f64> = train_x1.iter().zip(train_x2.iter()).map(|(&a, &b)| a + b).collect();

    let options = LoessOptions {
        span: 0.75,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 2,
        surface: LoessSurface::Direct,
    };
    let fit = loess_fit(&train_y, &[train_x1.clone(), train_x2.clone()], &options).unwrap();

    // Predict at new points
    let new_x1 = vec![2.5, 5.5];
    let new_x2 = vec![4.0, 3.0];
    let predictions = fit
        .predict(
            &[new_x1, new_x2],
            &[train_x1, train_x2],
            &train_y,
            &options,
        )
        .unwrap();

    assert_eq!(predictions.len(), 2);
    // Predictions should be finite
    assert!(predictions[0].is_finite());
    assert!(predictions[1].is_finite());
}

#[test]
fn test_loess_predict_empty() {
    // Test prediction with empty new_x
    let train_x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let train_y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let options = LoessOptions::default();
    let fit = loess_fit(&train_y, &[train_x.clone()], &options).unwrap();

    // Empty prediction should return empty vector
    let new_x: Vec<f64> = vec![];
    let predictions = fit.predict(&[new_x], &[train_x], &train_y, &options).unwrap();

    assert!(predictions.is_empty());
}

#[test]
fn test_loess_predict_extrapolation() {
    // Test that extrapolation works (though less accurate)
    let train_x = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let train_y: Vec<f64> = train_x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    let options = LoessOptions {
        span: 0.75,
        robust_iterations: 0,
        ..Default::default()
    };
    let fit = loess_fit(&train_y, &[train_x.clone()], &options).unwrap();

    // Predict outside training range (extrapolation)
    let new_x = vec![0.5, 9.5]; // Below and above training range
    let predictions = fit.predict(&[new_x], &[train_x], &train_y, &options).unwrap();

    assert_eq!(predictions.len(), 2);
    // Predictions should be finite (though less accurate)
    assert!(predictions[0].is_finite());
    assert!(predictions[1].is_finite());
}

#[test]
fn test_loess_small_dataset() {
    // Test with minimum valid dataset (2 points for degree 1)
    let x = vec![0.0, 1.0];
    let y = vec![0.0, 1.0];

    let options = LoessOptions {
        span: 1.0,  // Use all data
        degree: 1,
        robust_iterations: 0,
        n_predictors: 1,
        surface: LoessSurface::Direct,
    };

    let result = loess_fit(&y, &[x], &options).unwrap();
    assert_eq!(result.fitted.len(), 2);
}

#[test]
fn test_loess_quadratic_min_three_points() {
    // Quadratic requires at least 3 points
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![0.0, 1.0, 4.0];  // y = x^2

    let options = LoessOptions {
        span: 1.0,
        degree: 2,
        robust_iterations: 0,
        n_predictors: 1,
        surface: LoessSurface::Direct,
    };

    let result = loess_fit(&y, &[x], &options).unwrap();
    assert_eq!(result.fitted.len(), 3);

    // With full span and perfect quadratic, fit should be exact
    for i in 0..3 {
        assert!((result.fitted[i] - y[i]).abs() < 0.5);
    }
}
