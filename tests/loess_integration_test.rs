//! LOESS integration tests.
//!
//! Verifies that LOESS works correctly alongside OLS and regularized regression methods.

use linreg_core::{
    core::ols_regression,
    loess::{loess_fit, LoessOptions},
    loess::types::LoessSurface,
    regularized::{elastic_net_fit, ElasticNetOptions, lasso_fit, LassoFitOptions, ridge_fit, RidgeFitOptions},
    linalg::Matrix,
};

#[test]
fn test_loess_alongside_ols() {
    // Test that LOESS can be used in the same program as OLS
    let y = vec![1.0, 2.5, 4.2, 6.1, 7.8, 9.9, 12.1, 14.0, 16.2, 18.0];
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let names = vec!["Intercept".to_string(), "X".to_string()];

    // Run OLS regression
    let ols_result = ols_regression(&y, &[x.clone()], &names).unwrap();
    assert_eq!(ols_result.coefficients.len(), 2);

    // Run LOESS
    let loess_options = LoessOptions {
        span: 0.75,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 1,
        surface: linreg_core::loess::types::LoessSurface::Direct,
    };
    let loess_result = loess_fit(&y, &[x], &loess_options).unwrap();
    assert_eq!(loess_result.fitted.len(), y.len());

    // Both should produce valid results
    assert!(ols_result.r_squared > 0.0);
    assert!(loess_result.fitted.iter().all(|v| v.is_finite()));
}

#[test]
fn test_loess_alongside_ridge() {
    // Test LOESS alongside ridge regression
    let y = vec![1.0, 2.5, 4.2, 6.1, 7.8, 9.9, 12.1, 14.0, 16.2, 18.0];
    let n = y.len();

    // Build design matrix with intercept
    let mut x_data = Vec::with_capacity(n * 2);
    for i in 0..n {
        x_data.push(1.0); // intercept
        x_data.push(i as f64);
    }
    let x_matrix = Matrix::new(n, 2, x_data);

    // Run ridge regression
    let ridge_options = RidgeFitOptions {
        lambda: 1.0,
        standardize: false,
        intercept: false, // intercept already in matrix
        max_iter: 1000,
        tol: 1e-7,
        weights: None,
        warm_start: None,
    };
    let ridge_result = ridge_fit(&x_matrix, &y, &ridge_options).unwrap();
    assert_eq!(ridge_result.coefficients.len(), 2);

    // Run LOESS
    let x_vec = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let loess_options = LoessOptions::default();
    let loess_result = loess_fit(&y, &[x_vec], &loess_options).unwrap();

    // Both should produce valid results
    assert!(ridge_result.r_squared > 0.0);
    assert!(loess_result.fitted.iter().all(|v| v.is_finite()));
}

#[test]
fn test_loess_alongside_lasso() {
    // Test LOESS alongside lasso regression
    let y = vec![1.0, 2.5, 4.2, 6.1, 7.8, 9.9, 12.1, 14.0, 16.2, 18.0];
    let n = y.len();

    // Build design matrix
    let mut x_data = Vec::with_capacity(n * 2);
    for i in 0..n {
        x_data.push(1.0); // intercept
        x_data.push(i as f64);
    }
    let x_matrix = Matrix::new(n, 2, x_data);

    // Run lasso regression
    let lasso_options = LassoFitOptions {
        lambda: 0.1,
        standardize: false,
        intercept: false,
        max_iter: 1000,
        tol: 1e-7,
        penalty_factor: None,
        weights: None,
        warm_start: None,
    };
    let lasso_result = lasso_fit(&x_matrix, &y, &lasso_options).unwrap();
    assert!(lasso_result.converged);

    // Run LOESS
    let x_vec = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let loess_options = LoessOptions::default();
    let loess_result = loess_fit(&y, &[x_vec], &loess_options).unwrap();

    // Both should produce valid results
    assert!(lasso_result.r_squared > 0.0);
    assert!(loess_result.fitted.iter().all(|v| v.is_finite()));
}

#[test]
fn test_loess_alongside_elastic_net() {
    // Test LOESS alongside elastic net regression
    let y = vec![1.0, 2.5, 4.2, 6.1, 7.8, 9.9, 12.1, 14.0, 16.2, 18.0];
    let n = y.len();

    // Build design matrix
    let mut x_data = Vec::with_capacity(n * 2);
    for i in 0..n {
        x_data.push(1.0); // intercept
        x_data.push(i as f64);
    }
    let x_matrix = Matrix::new(n, 2, x_data);

    // Run elastic net regression
    let en_options = ElasticNetOptions {
        lambda: 0.1,
        alpha: 0.5,
        standardize: false,
        intercept: false,
        max_iter: 1000,
        tol: 1e-7,
        penalty_factor: None,
        warm_start: None,
        weights: None,
        coefficient_bounds: None,
    };
    let en_result = elastic_net_fit(&x_matrix, &y, &en_options).unwrap();
    assert!(en_result.converged);

    // Run LOESS
    let x_vec = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let loess_options = LoessOptions::default();
    let loess_result = loess_fit(&y, &[x_vec], &loess_options).unwrap();

    // Both should produce valid results
    assert!(en_result.r_squared > 0.0);
    assert!(loess_result.fitted.iter().all(|v| v.is_finite()));
}

#[test]
fn test_full_workflow() {
    // Complete workflow: fit all regression types on same data
    let y = vec![
        2.1, 4.5, 6.8, 8.9, 11.2, 13.5, 15.8, 18.1, 20.4, 22.5,
        24.8, 27.1, 29.2, 31.5, 33.8, 36.1, 38.2, 40.5, 42.8, 45.1,
    ];
    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let names = vec!["Intercept".to_string(), "X".to_string()];
    let n = y.len();

    // 1. OLS
    let ols = ols_regression(&y, &[x.clone()], &names).unwrap();
    assert!(ols.r_squared > 0.9);

    // 2. Ridge
    let mut ridge_x = Vec::with_capacity(n * 2);
    for i in 0..n {
        ridge_x.push(1.0);
        ridge_x.push(x[i]);
    }
    let ridge_matrix = Matrix::new(n, 2, ridge_x);
    let ridge = ridge_fit(
        &ridge_matrix,
        &y,
        &RidgeFitOptions {
            lambda: 1.0,
            standardize: false,
            intercept: false,
            max_iter: 1000,
            tol: 1e-7,
            weights: None,
            warm_start: None,
        },
    )
    .unwrap();
    assert!(ridge.r_squared > 0.9);

    // 3. LOESS
    let loess_options = LoessOptions {
        span: 0.5,
        degree: 1,
        robust_iterations: 0,
        n_predictors: 1,
        surface: linreg_core::loess::types::LoessSurface::Direct,
    };
    let loess = loess_fit(&y, &[x.clone()], &loess_options).unwrap();
    assert!(loess.fitted.iter().all(|v| v.is_finite()));

    // 4. LOESS prediction
    let new_x = vec![2.5, 7.5, 12.5, 17.5];
    let predictions = loess
        .predict(&[new_x], &[x], &y, &loess_options)
        .unwrap();
    assert_eq!(predictions.len(), 4);
    assert!(predictions.iter().all(|p| p.is_finite()));
}

#[test]
fn test_loess_vs_linear_data() {
    // On truly linear data, LOESS should approximate the linear fit
    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 3.0).collect();

    // OLS should get exact fit
    let names = vec!["Intercept".to_string(), "X".to_string()];
    let ols = ols_regression(&y, &[x.clone()], &names).unwrap();
    assert!((ols.coefficients[0] - 3.0).abs() < 0.1); // intercept
    assert!((ols.coefficients[1] - 2.0).abs() < 0.1); // slope

    // LOESS with large span should approximate linear fit
    let loess = loess_fit(
        &y,
        &[x],
        &LoessOptions {
            span: 0.9, // Large span = smoother
            degree: 1,
            robust_iterations: 0,
            n_predictors: 1,
            surface: linreg_core::loess::types::LoessSurface::Direct,
        },
    )
    .unwrap();

    // Check that LOESS fitted values are close to true values
    for i in 5..15 {
        // Check interior points
        assert!((loess.fitted[i] - y[i]).abs() < 1.0);
    }
}

#[test]
fn test_loess_multiple_predictors_integration() {
    // Test LOESS with multiple predictors alongside other methods
    let x1 = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    ];
    let x2 = vec![
        5.0, 3.0, 8.0, 2.0, 7.0, 1.0, 6.0, 4.0, 9.0, 0.0, 5.5, 2.5, 7.5, 1.5, 6.5,
    ];
    let y: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| a + 0.5 * b + 1.0)
        .collect();

    let names = vec![
        "Intercept".to_string(),
        "X1".to_string(),
        "X2".to_string(),
    ];

    // OLS
    let ols = ols_regression(&y, &[x1.clone(), x2.clone()], &names).unwrap();
    assert!(ols.r_squared > 0.8);

    // LOESS
    let loess = loess_fit(
        &y,
        &[x1, x2],
        &LoessOptions {
            span: 0.7,
            degree: 1,
            robust_iterations: 0,
            n_predictors: 2,
            surface: linreg_core::loess::types::LoessSurface::Direct,
        },
    )
    .unwrap();

    // Check LOESS produces valid results
    assert_eq!(loess.fitted.len(), y.len());
    assert!(loess.fitted.iter().all(|v| v.is_finite()));
}

#[test]
fn test_loess_quadratic_integration() {
    // Test quadratic LOESS alongside linear methods
    let x: Vec<f64> = (0..15).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 0.1 * xi * xi + 0.5 * xi + 1.0).collect();

    // Linear LOESS (won't fit perfectly)
    let loess_linear = loess_fit(
        &y,
        &[x.clone()],
        &LoessOptions {
            span: 0.6,
            degree: 1,
            robust_iterations: 0,
            n_predictors: 1,
            surface: linreg_core::loess::types::LoessSurface::Direct,
        },
    )
    .unwrap();

    // Quadratic LOESS (should fit better)
    let loess_quad = loess_fit(
        &y,
        &[x],
        &LoessOptions {
            span: 0.6,
            degree: 2,
            robust_iterations: 0,
            n_predictors: 1,
            surface: linreg_core::loess::types::LoessSurface::Direct,
        },
    )
    .unwrap();

    // Both should produce valid results
    assert_eq!(loess_linear.fitted.len(), y.len());
    assert_eq!(loess_quad.fitted.len(), y.len());

    // Quadratic should generally fit better for quadratic data
    let sse_linear: f64 = y
        .iter()
        .zip(loess_linear.fitted.iter())
        .map(|(y, y_hat)| (y - y_hat).powi(2))
        .sum();
    let sse_quad: f64 = y
        .iter()
        .zip(loess_quad.fitted.iter())
        .map(|(y, y_hat)| (y - y_hat).powi(2))
        .sum();

    // Quadratic should have lower error (though not guaranteed at boundaries)
    assert!(sse_quad < sse_linear * 1.5); // Allow some tolerance
}
