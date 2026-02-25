// FFI tests for polynomial regression functions.

#[cfg(feature = "ffi")]

use super::common::*;

// ============================================================================
// Polynomial Regression Tests
// ============================================================================

#[test]
fn test_polynomial_fit_basic() {
    // Create quadratic data: y = 1 + 2*x + 0.5*x^2
    let y: Vec<f64> = vec
![1.0, 3.5, 7.0, 11.5, 17.0, 23.5, 31.0, 39.5, 49.0, 59.5];
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();

    let degree = 2;
    let center = 0; // no centering

    let handle = unsafe {
        LR_PolynomialFit(
            y.as_ptr(),
            y.len() as i32,
            x.as_ptr(),
            degree,
            center,
        )
    };
    let _guard = HandleGuard::new(handle);

    assert!(handle > 0, "Handle should be positive");

    // Check degree
    let result_degree = unsafe { LR_GetPolynomialDegree(handle) };
    assert_eq!(result_degree, degree as i32);

    // Check center flag
    let result_center = unsafe { LR_GetPolynomialCenter(handle) };
    assert_eq!(result_center, center);

    // Check center value (should be 0 when not centered)
    let center_val = unsafe { LR_GetPolynomialCenterValue(handle) };
    assert_eq!(center_val, 0.0);

    // Check R-squared
    let r2 = unsafe { LR_GetPolynomialRSquared(handle) };
    assert!(r2 > 0.99, "R-squared should be high for perfect quadratic data");

    // Get coefficients
    let n_coef = unsafe { LR_GetPolynomialNumCoefficients(handle) } as usize;
    assert_eq!(n_coef, (degree + 1) as usize, "Should have degree + 1 coefficients");

    let mut coefs = vec![0.0; n_coef];
    unsafe {
        LR_GetPolynomialCoefficients(
            handle,
            coefs.as_mut_ptr(),
            n_coef as i32,
        );
    }

    // Coefficients should be close to [1.0, 2.0, 0.5]
    assert!((coefs[0] - 1.0).abs() < 0.1, "Intercept should be ~1.0");
    assert!((coefs[1] - 2.0).abs() < 0.1, "Linear coef should be ~2.0");
    assert!((coefs[2] - 0.5).abs() < 0.1, "Quadratic coef should be ~0.5");
}

#[test]
fn test_polynomial_fit_with_centering() {
    // Create quadratic data
    let y: Vec<f64> = vec
![1.0, 3.5, 7.0, 11.5, 17.0, 23.5, 31.0, 39.5, 49.0, 59.5];
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();

    let degree = 2;
    let center = 1; // enable centering

    let handle = unsafe {
        LR_PolynomialFit(
            y.as_ptr(),
            y.len() as i32,
            x.as_ptr(),
            degree,
            center,
        )
    };
    let _guard = HandleGuard::new(handle);

    assert!(handle > 0);

    // Check center flag
    let result_center = unsafe { LR_GetPolynomialCenter(handle) };
    assert_eq!(result_center, 1 as i32);

    // Check center value (should be the mean of x)
    let center_val = unsafe { LR_GetPolynomialCenterValue(handle) };
    let expected_mean = x.iter().sum::<f64>() / x.len() as f64;
    assert!((center_val - expected_mean).abs() < 0.01);

    // R-squared should still be good
    let r2 = unsafe { LR_GetPolynomialRSquared(handle) };
    assert!(r2 > 0.99);
}

#[test]
fn test_polynomial_predict() {
    // Create simple linear data for prediction test
    let y: Vec<f64> = vec
![2.0, 4.0, 6.0, 8.0, 10.0];
    let x: Vec<f64> = vec
![1.0, 2.0, 3.0, 4.0, 5.0];

    let degree = 1; // linear
    let center = 0;

    let handle = unsafe {
        LR_PolynomialFit(
            y.as_ptr(),
            y.len() as i32,
            x.as_ptr(),
            degree,
            center,
        )
    };
    let _guard = HandleGuard::new(handle);

    // Predict at new points
    let x_new = vec
![6.0, 7.0];
    let mut predictions = vec![0.0; x_new.len()];

    let n_written = unsafe {
        LR_PolynomialPredict(
            handle,
            x_new.as_ptr(),
            x_new.len() as i32,
            predictions.as_mut_ptr(),
        )
    };

    assert_eq!(n_written, x_new.len() as i32);

    // Predictions should be close to [12.0, 14.0] for y = 2*x
    for (i, &pred) in predictions.iter().enumerate() {
        let expected = 2.0 * x_new[i];
        assert!((pred - expected).abs() < 0.1, "Prediction at x={} should be ~{}", x_new[i], expected);
    }
}

#[test]
fn test_polynomial_predict_centered() {
    // Create quadratic data
    let y: Vec<f64> = (1..=10).map(|i| {
        let x = i as f64;
        1.0 + 2.0 * x + 0.5 * x * x
    }).collect();

    let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();

    let degree = 2;
    let center = 1; // centered

    let handle = unsafe {
        LR_PolynomialFit(
            y.as_ptr(),
            y.len() as i32,
            x.as_ptr(),
            degree,
            center,
        )
    };
    let _guard = HandleGuard::new(handle);

    // Predict at new points outside training range
    let x_new = vec
![15.0, 20.0];
    let mut predictions = vec![0.0; x_new.len()];

    let n_written = unsafe {
        LR_PolynomialPredict(
            handle,
            x_new.as_ptr(),
            x_new.len() as i32,
            predictions.as_mut_ptr(),
        )
    };

    assert_eq!(n_written, x_new.len() as i32);

    // Check predictions are reasonable (quadratic grows fast)
    let expected_15 = 1.0 + 2.0 * 15.0 + 0.5 * 15.0 * 15.0; // = 139.5
    let expected_20 = 1.0 + 2.0 * 20.0 + 0.5 * 20.0 * 20.0; // = 241.0

    assert!((predictions[0] - expected_15).abs() < 1.0);
    assert!((predictions[1] - expected_20).abs() < 1.0);
}

#[test]
fn test_polynomial_invalid_inputs() {
    let y: Vec<f64> = vec
![1.0, 2.0, 3.0];
    let x: Vec<f64> = vec
![1.0, 2.0, 3.0];

    // Test null pointer (simulated with empty data check in function)
    // Test with degree < 1
    let handle = unsafe {
        LR_PolynomialFit(
            y.as_ptr(),
            y.len() as i32,
            x.as_ptr(),
            0, // invalid degree
            0,
        )
    };

    assert_eq!(handle, 0, "Invalid degree should return handle 0");
}

#[test]
fn test_polynomial_stats() {
    // Create cubic data
    let y: Vec<f64> = (0..10).map(|i| {
        let x = i as f64;
        5.0 - 2.0 * x + 0.5 * x * x - 0.05 * x * x * x
    }).collect();

    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();

    let degree = 3;
    let center = 1; // centering recommended for cubic

    let handle = unsafe {
        LR_PolynomialFit(
            y.as_ptr(),
            y.len() as i32,
            x.as_ptr(),
            degree,
            center,
        )
    };
    let _guard = HandleGuard::new(handle);

    // Get all stats
    let r2 = unsafe { LR_GetPolynomialRSquared(handle) };
    let adj_r2 = unsafe { LR_GetPolynomialAdjRSquared(handle) };
    let mse = unsafe { LR_GetPolynomialMSE(handle) };

    assert!(r2.is_finite());
    assert!(adj_r2.is_finite());
    assert!(mse.is_finite());
    assert!(mse >= 0.0);
    assert!(r2 > 0.9, "Cubic fit should be good");
}

#[test]
fn test_polynomial_higher_degree() {
    // Test degree 4 polynomial
    let y: Vec<f64> = (0..15).map(|i| {
        let x = i as f64;
        10.0 + x + 0.1 * x * x + 0.01 * x * x * x + 0.001 * x * x * x * x
    }).collect();

    let x: Vec<f64> = (0..15).map(|i| i as f64).collect();

    let degree = 4;
    let center = 1;

    let handle = unsafe {
        LR_PolynomialFit(
            y.as_ptr(),
            y.len() as i32,
            x.as_ptr(),
            degree,
            center,
        )
    };
    let _guard = HandleGuard::new(handle);

    assert!(handle > 0);

    let n_coef = unsafe { LR_GetPolynomialNumCoefficients(handle) } as usize;
    assert_eq!(n_coef, 5); // degree + 1

    // Get coefficients
    let mut coefs = vec![0.0; n_coef];
    unsafe {
        LR_GetPolynomialCoefficients(
            handle,
            coefs.as_mut_ptr(),
            n_coef as i32,
        );
    }

    // All coefficients should be finite
    for coef in &coefs {
        assert!(coef.is_finite());
    }
}
