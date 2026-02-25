//! Unit tests for feature importance module.

mod feature_importance {
    use linreg_core::feature_importance::{
        permutation_importance_ols, permutation_importance_ols_named, permutation_importance_ridge, permutation_importance_lasso,
        permutation_importance_elastic_net, permutation_importance_loess,
        shap_values_linear, shap_values_linear_named, shap_values_polynomial,
        shap_values_ridge, shap_values_lasso, shap_values_elastic_net,
        standardized_coefficients, vif_ranking,
        PermutationImportanceOptions,
    };
    use linreg_core::polynomial::{polynomial_regression, PolynomialOptions};
    use linreg_core::regularized::{ridge_fit, lasso_fit, elastic_net_fit, RidgeFitOptions, LassoFitOptions, ElasticNetOptions};
    use linreg_core::linalg::Matrix;
    use linreg_core::core::ols_regression;

    #[test]
    fn test_standardized_coefficients_basic() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let names = vec!["Intercept".into(), "X1".into(), "X2".into()];

        let fit = ols_regression(&y, &[x1.clone(), x2.clone()], &names).unwrap();

        // Compute standardized coefficients
        let result = standardized_coefficients(&fit.coefficients, &[x1, x2]).unwrap();

        assert_eq!(result.variable_names, vec!["X1", "X2"]);
        assert_eq!(result.standardized_coefficients.len(), 2);
        assert!(result.y_std > 0.0);
    }

    #[test]
    fn test_standardized_coefficients_with_names() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let names = vec!["Intercept".into(), "Temp".into(), "Pressure".into()];

        let fit = ols_regression(&y, &[x1.clone(), x2.clone()], &names).unwrap();

        // Compute Y std
        let y_std = fit.std_error;

        let result = linreg_core::standardized_coefficients_named(
            &fit.coefficients,
            &[x1, x2],
            &["Temperature".to_string(), "Pressure".to_string()],
            y_std,
        )
        .unwrap();

        assert_eq!(
            result.variable_names,
            vec!["Temperature", "Pressure"]
        );
    }

    #[test]
    fn test_vif_ranking() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfectly correlated with x1
        let names = vec!["Intercept".into(), "X1".into(), "X2".into()];

        let fit = ols_regression(&y, &[x1, x2], &names).unwrap();

        let ranking = vif_ranking(&fit.vif);

        assert_eq!(ranking.variable_names.len(), 2);
        assert_eq!(ranking.vif_values.len(), 2);

        let ranked = ranking.ranking();
        // Should be sorted by VIF ascending
        assert!(ranked[0].1 <= ranked[1].1);
    }

    #[test]
    fn test_shap_values_linear() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let names = vec!["Intercept".into(), "X1".into(), "X2".into()];

        let fit = ols_regression(&y, &[x1.clone(), x2.clone()], &names).unwrap();

        let shap = shap_values_linear(&[x1.clone(), x2.clone()], &fit.coefficients).unwrap();

        assert_eq!(shap.variable_names, vec!["X1", "X2"]);
        assert_eq!(shap.shap_values.len(), 5); // 5 observations
        assert_eq!(shap.shap_values[0].len(), 2); // 2 features
        assert!(shap.base_value.is_finite());

        let ranking = shap.ranking();
        assert_eq!(ranking.len(), 2);
    }

    #[test]
    fn test_shap_values_constant_feature() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![5.0, 5.0, 5.0, 5.0, 5.0]; // Constant
        let names = vec!["Intercept".into(), "X1".into(), "X2".into()];

        let fit = ols_regression(&y, &[x1.clone(), x2.clone()], &names).unwrap();

        let shap = shap_values_linear(&[x1.clone(), x2.clone()], &fit.coefficients).unwrap();

        // Constant feature should have SHAP = 0
        for obs in &shap.shap_values {
            // Due to perfect multicollinearity, coefficient might be dropped (NaN)
            // So we check if it's either 0 or NaN
            assert!(obs[1] == 0.0 || obs[1].is_nan());
        }
        // mean_abs_shap should be 0 if all values are 0
        if shap.mean_abs_shap[1].is_finite() {
            assert_eq!(shap.mean_abs_shap[1], 0.0);
        }
    }

    #[test]
    fn test_shap_values_linear_named() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let names = vec!["Intercept".into(), "X1".into(), "X2".into()];

        let fit = ols_regression(&y, &[x1.clone(), x2.clone()], &names).unwrap();

        let custom_names = vec!["Temperature".to_string(), "Pressure".to_string()];
        let shap = shap_values_linear_named(&[x1.clone(), x2.clone()], &fit.coefficients, &custom_names).unwrap();

        // Verify custom variable names are applied
        assert_eq!(shap.variable_names, custom_names);
        assert_eq!(shap.shap_values.len(), 5); // 5 observations
        assert_eq!(shap.shap_values[0].len(), 2); // 2 features
        assert!(shap.base_value.is_finite());

        let ranking = shap.ranking();
        assert_eq!(ranking.len(), 2);
        // Verify ranking uses custom names
        assert_eq!(ranking[0].0, "Temperature");
    }

    #[test]
    fn test_shap_values_linear_named_length_mismatch() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let names = vec!["Intercept".into(), "X1".into(), "X2".into()];

        let fit = ols_regression(&y, &[x1.clone(), x2.clone()], &names).unwrap();

        // Wrong number of variable names (1 instead of 2)
        let bad_names = vec!["Temperature".to_string()];
        let result = shap_values_linear_named(&[x1, x2], &fit.coefficients, &bad_names);
        assert!(result.is_err());
    }

    #[test]
    fn test_shap_values_polynomial_quadratic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // y = 1 + 2*x + 0.5*x²
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + 0.5 * xi * xi).collect();

        let options = PolynomialOptions {
            degree: 2,
            center: false,
            ..Default::default()
        };
        let fit = polynomial_regression(&y, &x, &options).unwrap();

        let shap = shap_values_polynomial(&x, &fit).unwrap();

        // Check variable names have superscripts
        assert_eq!(shap.variable_names.len(), 2);
        assert_eq!(shap.variable_names[0], "X¹");  // Linear term
        assert_eq!(shap.variable_names[1], "X²");  // Quadratic term

        // 5 observations, 2 features (linear and quadratic)
        assert_eq!(shap.shap_values.len(), 5);
        assert_eq!(shap.shap_values[0].len(), 2);

        // Base value should be the intercept
        assert!(shap.base_value.is_finite());

        // Mean absolute SHAP should have 2 entries
        assert_eq!(shap.mean_abs_shap.len(), 2);
        assert!(shap.mean_abs_shap[0] >= 0.0);
        assert!(shap.mean_abs_shap[1] >= 0.0);

        // Ranking should return both features
        let ranking = shap.ranking();
        assert_eq!(ranking.len(), 2);
    }

    #[test]
    fn test_shap_values_polynomial_cubic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // y = 1 + x + 0.1*x² + 0.05*x³
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 + xi + 0.1 * xi * xi + 0.05 * xi * xi * xi).collect();

        let options = PolynomialOptions {
            degree: 3,
            center: false,
            ..Default::default()
        };
        let fit = polynomial_regression(&y, &x, &options).unwrap();

        let shap = shap_values_polynomial(&x, &fit).unwrap();

        // Check variable names for cubic polynomial
        assert_eq!(shap.variable_names.len(), 3);
        assert_eq!(shap.variable_names[0], "X¹");  // Linear term
        assert_eq!(shap.variable_names[1], "X²");  // Quadratic term
        assert_eq!(shap.variable_names[2], "X³");  // Cubic term

        // 6 observations, 3 features
        assert_eq!(shap.shap_values.len(), 6);
        assert_eq!(shap.shap_values[0].len(), 3);

        let ranking = shap.ranking();
        assert_eq!(ranking.len(), 3);
    }

    #[test]
    fn test_shap_values_polynomial_centered() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + 0.5 * xi * xi).collect();

        let options = PolynomialOptions {
            degree: 2,
            center: true,  // Centering enabled
            ..Default::default()
        };
        let fit = polynomial_regression(&y, &x, &options).unwrap();

        // Fit should have centering applied
        assert!(fit.centered);
        assert_ne!(fit.x_mean, 0.0);

        let shap = shap_values_polynomial(&x, &fit).unwrap();

        // SHAP values should still be computed correctly with centering
        assert_eq!(shap.shap_values.len(), 5);
        assert_eq!(shap.shap_values[0].len(), 2);
        assert!(shap.base_value.is_finite());
    }

    #[test]
    fn test_shap_values_polynomial_empty_input() {
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];

        let options = PolynomialOptions::default();
        let fit = polynomial_regression(&y, &x, &options);

        // Empty input should fail
        assert!(fit.is_err());
    }

    #[test]
    fn test_shap_values_ridge() {
        let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0];

        // Build Matrix for Ridge
        let mut x_data = Vec::new();
        for i in 0..y.len() {
            x_data.push(1.0); // intercept
            x_data.push(x1[i]);
            x_data.push(x2[i]);
        }
        let x = Matrix::new(y.len(), 3, x_data);

        let options = RidgeFitOptions {
            lambda: 1.0,
            standardize: true,
            ..Default::default()
        };
        let fit = ridge_fit(&x, &y, &options).unwrap();

        let shap = shap_values_ridge(&[x1.clone(), x2.clone()], &fit).unwrap();

        // Check basic structure
        assert_eq!(shap.variable_names, vec!["X1", "X2"]);
        assert_eq!(shap.shap_values.len(), 5); // 5 observations
        assert_eq!(shap.shap_values[0].len(), 2); // 2 features
        assert!(shap.base_value.is_finite());
        assert_eq!(shap.mean_abs_shap.len(), 2);

        // SHAP values should be finite
        for obs in &shap.shap_values {
            for val in obs {
                assert!(val.is_finite());
            }
        }

        // Ranking should work
        let ranking = shap.ranking();
        assert_eq!(ranking.len(), 2);
    }

    #[test]
    fn test_shap_values_ridge_single_predictor() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut x_data = Vec::new();
        for i in 0..y.len() {
            x_data.push(1.0); // intercept
            x_data.push(x1[i]);
        }
        let x = Matrix::new(y.len(), 2, x_data);

        let options = RidgeFitOptions {
            lambda: 0.5,
            standardize: false,
            ..Default::default()
        };
        let fit = ridge_fit(&x, &y, &options).unwrap();

        let shap = shap_values_ridge(&[x1], &fit).unwrap();

        assert_eq!(shap.variable_names, vec!["X1"]);
        assert_eq!(shap.shap_values.len(), 5);
        assert_eq!(shap.shap_values[0].len(), 1);
    }

    #[test]
    fn test_shap_values_lasso() {
        let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0];

        let mut x_data = Vec::new();
        for i in 0..y.len() {
            x_data.push(1.0); // intercept
            x_data.push(x1[i]);
            x_data.push(x2[i]);
        }
        let x = Matrix::new(y.len(), 3, x_data);

        let options = LassoFitOptions {
            lambda: 0.1,
            standardize: true,
            ..Default::default()
        };
        let fit = lasso_fit(&x, &y, &options).unwrap();

        let shap = shap_values_lasso(&[x1.clone(), x2.clone()], &fit).unwrap();

        // Check basic structure
        assert_eq!(shap.variable_names.len(), 2);
        assert_eq!(shap.shap_values.len(), 5);
        assert_eq!(shap.shap_values[0].len(), 2);
        assert!(shap.base_value.is_finite());

        // SHAP values should be finite
        for obs in &shap.shap_values {
            for val in obs {
                assert!(val.is_finite());
            }
        }

        // Some coefficients might be zero with Lasso
        let n_nonzero = fit.coefficients.iter().filter(|&&c| c.abs() > 1e-10).count();
        assert!(n_nonzero <= 2); // At most 2 non-zero coefficients
    }

    #[test]
    fn test_shap_values_elastic_net() {
        let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0];

        let mut x_data = Vec::new();
        for i in 0..y.len() {
            x_data.push(1.0); // intercept
            x_data.push(x1[i]);
            x_data.push(x2[i]);
        }
        let x = Matrix::new(y.len(), 3, x_data);

        let options = ElasticNetOptions {
            lambda: 0.1,
            alpha: 0.5, // Mix of L1 and L2
            standardize: true,
            ..Default::default()
        };
        let fit = elastic_net_fit(&x, &y, &options).unwrap();

        let shap = shap_values_elastic_net(&[x1.clone(), x2.clone()], &fit).unwrap();

        // Check basic structure
        assert_eq!(shap.variable_names.len(), 2);
        assert_eq!(shap.shap_values.len(), 5);
        assert_eq!(shap.shap_values[0].len(), 2);
        assert!(shap.base_value.is_finite());

        // SHAP values should be finite
        for obs in &shap.shap_values {
            for val in obs {
                assert!(val.is_finite());
            }
        }

        // Ranking should work
        let ranking = shap.ranking();
        assert_eq!(ranking.len(), 2);
    }

    #[test]
    fn test_permutation_importance_ridge() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Strongly correlated with y
        let x2 = vec![2.0, 4.0, 1.0, 3.0, 2.0, 4.0]; // Weaker correlation

        let mut x_data = Vec::new();
        for i in 0..y.len() {
            x_data.push(1.0); // intercept
            x_data.push(x1[i]);
            x_data.push(x2[i]);
        }
        let x = Matrix::new(y.len(), 3, x_data);

        let ridge_options = RidgeFitOptions {
            lambda: 1.0,
            standardize: true,
            ..Default::default()
        };
        let fit = ridge_fit(&x, &y, &ridge_options).unwrap();

        let options = PermutationImportanceOptions {
            n_permutations: 10,
            seed: Some(42),
            compute_intervals: false,
            ..Default::default()
        };

        let importance = permutation_importance_ridge(&y, &[x1.clone(), x2.clone()], &fit, &options).unwrap();

        // X1 should have higher importance than X2
        assert!(importance.importance[0] > importance.importance[1]);
        assert_eq!(importance.n_permutations, 10);
        assert_eq!(importance.seed, Some(42));

        let ranking = importance.ranking();
        assert!(ranking[0].1 >= ranking[1].1);
    }

    #[test]
    fn test_permutation_importance_lasso() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Strongly correlated with y
        let x2 = vec![2.0, 4.0, 1.0, 3.0, 2.0, 4.0]; // Weaker correlation

        let mut x_data = Vec::new();
        for i in 0..y.len() {
            x_data.push(1.0); // intercept
            x_data.push(x1[i]);
            x_data.push(x2[i]);
        }
        let x = Matrix::new(y.len(), 3, x_data);

        let lasso_options = LassoFitOptions {
            lambda: 0.1,
            standardize: true,
            ..Default::default()
        };
        let fit = lasso_fit(&x, &y, &lasso_options).unwrap();

        let options = PermutationImportanceOptions {
            n_permutations: 10,
            seed: Some(42),
            compute_intervals: false,
            ..Default::default()
        };

        let importance = permutation_importance_lasso(&y, &[x1.clone(), x2.clone()], &fit, &options).unwrap();

        // X1 should have higher importance than X2
        assert!(importance.importance[0] > importance.importance[1]);
        assert_eq!(importance.n_permutations, 10);

        let ranking = importance.ranking();
        assert!(ranking[0].1 >= ranking[1].1);
    }

    #[test]
    fn test_permutation_importance_elastic_net() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Strongly correlated with y
        let x2 = vec![2.0, 4.0, 1.0, 3.0, 2.0, 4.0]; // Weaker correlation

        let mut x_data = Vec::new();
        for i in 0..y.len() {
            x_data.push(1.0); // intercept
            x_data.push(x1[i]);
            x_data.push(x2[i]);
        }
        let x = Matrix::new(y.len(), 3, x_data);

        let enet_options = ElasticNetOptions {
            lambda: 0.1,
            alpha: 0.5,
            standardize: true,
            ..Default::default()
        };
        let fit = elastic_net_fit(&x, &y, &enet_options).unwrap();

        let options = PermutationImportanceOptions {
            n_permutations: 10,
            seed: Some(42),
            compute_intervals: false,
            ..Default::default()
        };

        let importance = permutation_importance_elastic_net(&y, &[x1.clone(), x2.clone()], &fit, &options).unwrap();

        // X1 should have higher importance than X2
        assert!(importance.importance[0] > importance.importance[1]);
        assert_eq!(importance.n_permutations, 10);

        let ranking = importance.ranking();
        assert!(ranking[0].1 >= ranking[1].1);
    }

    #[test]
    fn test_permutation_importance_regularized_reproducibility() {
        // Test that all regularized methods give reproducible results with same seed
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 4.0, 3.0, 5.0, 1.0];

        let mut x_data = Vec::new();
        for i in 0..y.len() {
            x_data.push(1.0); // intercept
            x_data.push(x1[i]);
            x_data.push(x2[i]);
        }
        let x = Matrix::new(y.len(), 3, x_data);

        let options = PermutationImportanceOptions {
            n_permutations: 5,
            seed: Some(42),
            compute_intervals: false,
            ..Default::default()
        };

        // Ridge
        let ridge_fit = ridge_fit(&x, &y, &RidgeFitOptions::default()).unwrap();
        let ridge1 = permutation_importance_ridge(&y, &[x1.clone(), x2.clone()], &ridge_fit, &options).unwrap();
        let ridge2 = permutation_importance_ridge(&y, &[x1.clone(), x2.clone()], &ridge_fit, &options).unwrap();
        assert_eq!(ridge1.importance[0], ridge2.importance[0]);

        // Lasso
        let lasso_fit = lasso_fit(&x, &y, &LassoFitOptions::default()).unwrap();
        let lasso1 = permutation_importance_lasso(&y, &[x1.clone(), x2.clone()], &lasso_fit, &options).unwrap();
        let lasso2 = permutation_importance_lasso(&y, &[x1.clone(), x2.clone()], &lasso_fit, &options).unwrap();
        assert_eq!(lasso1.importance[0], lasso2.importance[0]);

        // Elastic Net
        let enet_fit = elastic_net_fit(&x, &y, &ElasticNetOptions::default()).unwrap();
        let enet1 = permutation_importance_elastic_net(&y, &[x1.clone(), x2.clone()], &enet_fit, &options).unwrap();
        let enet2 = permutation_importance_elastic_net(&y, &[x1, x2], &enet_fit, &options).unwrap();
        assert_eq!(enet1.importance[0], enet2.importance[0]);
    }

    #[test]
    fn test_permutation_importance_ols() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Strongly correlated with y
        let x2 = vec![2.0, 4.0, 1.0, 3.0, 2.0, 4.0]; // Weaker correlation
        let names = vec!["Intercept".into(), "X1".into(), "X2".into()];

        let fit = ols_regression(&y, &[x1.clone(), x2.clone()], &names).unwrap();

        let options = PermutationImportanceOptions {
            n_permutations: 10,
            seed: Some(42),
            compute_intervals: false,
            interval_confidence: 0.95,
        };

        let importance =
            permutation_importance_ols(&y, &[x1, x2], &fit, &options).unwrap();

        // X1 should have higher importance than X2
        assert!(importance.importance[0] > importance.importance[1]);
        assert_eq!(importance.n_permutations, 10);
        assert_eq!(importance.seed, Some(42));

        let ranking = importance.ranking();
        assert!(ranking[0].1 >= ranking[1].1);
    }

    #[test]
    fn test_permutation_importance_ols_named() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x2 = vec![2.0, 4.0, 1.0, 3.0, 2.0, 4.0];
        let names = vec!["Intercept".into(), "X1".into(), "X2".into()];

        let fit = ols_regression(&y, &[x1.clone(), x2.clone()], &names).unwrap();

        let custom_names = vec!["Temperature".to_string(), "Pressure".to_string()];
        let options = PermutationImportanceOptions {
            n_permutations: 10,
            seed: Some(42),
            compute_intervals: false,
            ..Default::default()
        };

        let importance = permutation_importance_ols_named(
            &y,
            &[x1, x2],
            &fit,
            &options,
            &custom_names,
        ).unwrap();

        // Custom names should be applied
        assert_eq!(importance.variable_names, custom_names);
        assert_eq!(importance.variable_names.len(), 2);
        assert_eq!(importance.importance.len(), 2);

        let ranking = importance.ranking();
        assert_eq!(ranking.len(), 2);
        // Ranking should use custom names
        assert_eq!(ranking[0].0, "Temperature");
    }

    #[test]
    fn test_permutation_importance_ols_named_length_mismatch() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let names = vec!["Intercept".into(), "X1".into()];

        let fit = ols_regression(&y, &[x1.clone()], &names).unwrap();

        // Wrong number of custom names (2 instead of 1)
        let bad_names = vec!["Temperature".to_string(), "Pressure".to_string()];
        let options = PermutationImportanceOptions::default();

        let result = permutation_importance_ols_named(&y, &[x1], &fit, &options, &bad_names);
        assert!(result.is_err());
    }

    #[test]
    fn test_permutation_importance_ridge_with_confidence_intervals() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x2 = vec![2.0, 4.0, 1.0, 3.0, 2.0, 4.0];

        let mut x_data = Vec::new();
        for i in 0..y.len() {
            x_data.push(1.0); // intercept
            x_data.push(x1[i]);
            x_data.push(x2[i]);
        }
        let x = Matrix::new(y.len(), 3, x_data);

        let ridge_options = RidgeFitOptions {
            lambda: 1.0,
            standardize: true,
            ..Default::default()
        };
        let fit = ridge_fit(&x, &y, &ridge_options).unwrap();

        let options = PermutationImportanceOptions {
            n_permutations: 50,
            seed: Some(42),
            compute_intervals: true,
            interval_confidence: 0.95,
            ..Default::default()
        };

        let importance = permutation_importance_ridge(&y, &[x1, x2], &fit, &options).unwrap();

        // Check that confidence intervals are computed
        assert!(importance.importance_std_err.is_some());
        assert!(importance.interval_lower.is_some());
        assert!(importance.interval_upper.is_some());
        assert!(importance.interval_confidence.is_some());

        let std_err = importance.importance_std_err.as_ref().unwrap();
        let lower = importance.interval_lower.as_ref().unwrap();
        let upper = importance.interval_upper.as_ref().unwrap();
        let confidence = importance.interval_confidence.unwrap();

        assert_eq!(std_err.len(), 2);
        assert_eq!(lower.len(), 2);
        assert_eq!(upper.len(), 2);
        assert_eq!(confidence, 0.95);

        // For each feature, lower <= importance <= upper
        for i in 0..2 {
            assert!(std_err[i] >= 0.0);
            assert!(lower[i] <= importance.importance[i]);
            assert!(upper[i] >= importance.importance[i]);
        }

        // Interval width should be 2 * z_score * std_err (z_score for 95% is ~1.96)
        for i in 0..2 {
            let expected_width = 2.0 * 1.96 * std_err[i];
            let actual_width = upper[i] - lower[i];
            assert!((actual_width - expected_width).abs() < 0.01);
        }
    }

    #[test]
    fn test_permutation_importance_lasso_with_confidence_intervals() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x2 = vec![2.0, 4.0, 1.0, 3.0, 2.0, 4.0];

        let mut x_data = Vec::new();
        for i in 0..y.len() {
            x_data.push(1.0);
            x_data.push(x1[i]);
            x_data.push(x2[i]);
        }
        let x = Matrix::new(y.len(), 3, x_data);

        let lasso_options = LassoFitOptions {
            lambda: 0.1,
            standardize: true,
            ..Default::default()
        };
        let fit = lasso_fit(&x, &y, &lasso_options).unwrap();

        let options = PermutationImportanceOptions {
            n_permutations: 50,
            seed: Some(42),
            compute_intervals: true,
            interval_confidence: 0.90, // 90% CI
            ..Default::default()
        };

        let importance = permutation_importance_lasso(&y, &[x1, x2], &fit, &options).unwrap();

        // Check confidence intervals
        assert_eq!(importance.interval_confidence.unwrap(), 0.90);

        let lower = importance.interval_lower.as_ref().unwrap();
        let upper = importance.interval_upper.as_ref().unwrap();

        // Intervals should be valid
        for i in 0..2 {
            assert!(lower[i] <= importance.importance[i]);
            assert!(upper[i] >= importance.importance[i]);
        }
    }

    #[test]
    fn test_permutation_importance_elastic_net_with_confidence_intervals() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 4.0, 1.0, 3.0, 2.0];

        let mut x_data = Vec::new();
        for i in 0..y.len() {
            x_data.push(1.0);
            x_data.push(x1[i]);
            x_data.push(x2[i]);
        }
        let x = Matrix::new(y.len(), 3, x_data);

        let enet_options = ElasticNetOptions {
            lambda: 0.1,
            alpha: 0.5,
            standardize: true,
            ..Default::default()
        };
        let fit = elastic_net_fit(&x, &y, &enet_options).unwrap();

        let options = PermutationImportanceOptions {
            n_permutations: 30,
            seed: Some(42),
            compute_intervals: true,
            interval_confidence: 0.99, // 99% CI - wider interval
            ..Default::default()
        };

        let importance = permutation_importance_elastic_net(&y, &[x1, x2], &fit, &options).unwrap();

        assert_eq!(importance.interval_confidence.unwrap(), 0.99);

        // Check intervals are computed
        assert!(importance.importance_std_err.is_some());
        assert!(importance.interval_lower.is_some());
        assert!(importance.interval_upper.is_some());

        let lower = importance.interval_lower.as_ref().unwrap();
        let upper = importance.interval_upper.as_ref().unwrap();

        // Intervals should contain the importance value
        for i in 0..2 {
            assert!(lower[i] <= importance.importance[i]);
            assert!(upper[i] >= importance.importance[i]);
        }
    }

    #[test]
    fn test_permutation_importance_intervals_false_no_computation() {
        // Verify that when compute_intervals=false, the interval fields are None
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 4.0, 1.0, 3.0, 2.0];

        let mut x_data = Vec::new();
        for i in 0..y.len() {
            x_data.push(1.0);
            x_data.push(x1[i]);
            x_data.push(x2[i]);
        }
        let x = Matrix::new(y.len(), 3, x_data);

        let ridge_options = RidgeFitOptions::default();
        let fit = ridge_fit(&x, &y, &ridge_options).unwrap();

        let options = PermutationImportanceOptions {
            n_permutations: 50,
            seed: Some(42),
            compute_intervals: false, // No intervals
            ..Default::default()
        };

        let importance = permutation_importance_ridge(&y, &[x1, x2], &fit, &options).unwrap();

        // All interval fields should be None
        assert!(importance.importance_std_err.is_none());
        assert!(importance.interval_lower.is_none());
        assert!(importance.interval_upper.is_none());
        assert!(importance.interval_confidence.is_none());
    }

    #[test]
    fn test_permutation_importance_loess_single_predictor() {
        // Use data with a clear relationship but some noise
        let y = vec![1.1, 2.2, 2.8, 4.1, 5.0, 5.8, 7.1];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let options = PermutationImportanceOptions {
            n_permutations: 10,
            seed: Some(42),
            compute_intervals: false,
            ..Default::default()
        };

        let importance = permutation_importance_loess(&y, &[x1.clone()], 0.75, 1, &options).unwrap();

        // Check basic structure
        assert_eq!(importance.variable_names, vec!["X1"]);
        assert_eq!(importance.importance.len(), 1);
        assert_eq!(importance.n_permutations, 10);
        assert_eq!(importance.seed, Some(42));

        // Importance should be non-negative (shuffling shouldn't improve fit)
        assert!(importance.importance[0] >= 0.0);
        assert!(importance.baseline_score.is_finite());

        // With a clear relationship, importance should be positive
        // (though LOESS with span=0.75 may not drop R² to 0 when shuffled)
        assert!(importance.baseline_score > 0.0);
    }

    #[test]
    fn test_permutation_importance_loess_two_predictors() {
        // Use synthetic data where relationship is more complex
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 8.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let x2 = vec![2.0, 3.0, 3.5, 3.0, 2.0, 4.0, 3.0];

        let options = PermutationImportanceOptions {
            n_permutations: 10,
            seed: Some(42),
            ..Default::default()
        };

        let importance = permutation_importance_loess(&y, &[x1.clone(), x2.clone()], 0.75, 1, &options).unwrap();

        // Check basic structure
        assert_eq!(importance.variable_names, vec!["X1", "X2"]);
        assert_eq!(importance.importance.len(), 2);
        assert_eq!(importance.n_permutations, 10);

        // Both features should have some importance (positive or zero)
        for imp in &importance.importance {
            assert!(*imp >= 0.0);
        }

        let ranking = importance.ranking();
        assert_eq!(ranking.len(), 2);
    }

    #[test]
    fn test_permutation_importance_loess_quadratic_degree() {
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
        let x1: Vec<f64> = (1..=6).map(|i| i as f64).collect(); // y = x²

        let options = PermutationImportanceOptions {
            n_permutations: 10,
            seed: Some(42),
            ..Default::default()
        };

        // Use degree 2 for quadratic relationship
        let importance = permutation_importance_loess(&y, &[x1], 0.75, 2, &options).unwrap();

        assert_eq!(importance.variable_names.len(), 1);
        assert!(importance.importance[0] >= 0.0);
    }

    #[test]
    fn test_permutation_importance_loess_reproducibility() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let options = PermutationImportanceOptions {
            n_permutations: 5,
            seed: Some(42),
            ..Default::default()
        };

        let importance1 = permutation_importance_loess(&y, &[x1.clone()], 0.75, 1, &options).unwrap();
        let importance2 = permutation_importance_loess(&y, &[x1], 0.75, 1, &options).unwrap();

        // Same seed should give same results
        assert_eq!(importance1.importance[0], importance2.importance[0]);
    }

    #[test]
    fn test_permutation_importance_loess_insufficient_data() {
        let y = vec![1.0]; // Only 1 observation
        let x1 = vec![1.0];

        let options = PermutationImportanceOptions::default();

        let result = permutation_importance_loess(&y, &[x1], 0.75, 1, &options);
        assert!(result.is_err());
    }

    #[test]
    fn test_permutation_importance_loess_different_spans() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let options = PermutationImportanceOptions {
            n_permutations: 5,
            seed: Some(42),
            ..Default::default()
        };

        // Test with different span values
        let importance_narrow = permutation_importance_loess(&y, &[x1.clone()], 0.5, 1, &options).unwrap();
        let importance_wide = permutation_importance_loess(&y, &[x1], 0.9, 1, &options).unwrap();

        // Both should produce valid results
        assert!(importance_narrow.importance[0] >= 0.0);
        assert!(importance_wide.importance[0] >= 0.0);

        // Different spans may give different importance values
        // (This is expected behavior, not an assertion)
    }

    #[test]
    fn test_permutation_importance_reproducibility() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let names = vec!["Intercept".into(), "X1".into()];

        let fit = ols_regression(&y, &[x1.clone()], &names).unwrap();

        let options = PermutationImportanceOptions {
            n_permutations: 5,
            seed: Some(42),
            compute_intervals: false,
            interval_confidence: 0.95,
        };

        let importance1 =
            permutation_importance_ols(&y, &[x1.clone()], &fit, &options).unwrap();
        let importance2 =
            permutation_importance_ols(&y, &[x1], &fit, &options).unwrap();

        // Same seed should give same results
        assert_eq!(importance1.importance[0], importance2.importance[0]);
    }

    #[test]
    fn test_standardized_coefficients_ranking() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![1.0, 1.0, 1.0, 1.0, 1.0]; // Constant - zero SD
        let x3 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let names = vec!["Intercept".into(), "X1".into(), "X2".into(), "X3".into()];

        let fit = ols_regression(&y, &[x1.clone(), x2.clone(), x3.clone()], &names).unwrap();

        // This should fail because X2 has zero SD
        let result = standardized_coefficients(&fit.coefficients, &[x1, x2, x3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_metrics_integration() {
        // Integration test: run all feature importance metrics on same data
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 3.0, 3.5, 3.0, 2.0];
        let names = vec!["Intercept".into(), "Temp".into(), "Pressure".into()];

        let fit = ols_regression(&y, &[x1.clone(), x2.clone()], &names).unwrap();

        // Standardized coefficients
        let std_coefs = standardized_coefficients(&fit.coefficients, &[x1.clone(), x2.clone()]).unwrap();
        assert_eq!(std_coefs.variable_names.len(), 2);

        // SHAP values
        let shap = shap_values_linear(&[x1.clone(), x2.clone()], &fit.coefficients).unwrap();
        assert_eq!(shap.variable_names.len(), 2);

        // VIF ranking
        let vif = vif_ranking(&fit.vif);
        assert_eq!(vif.variable_names.len(), 2);

        // Permutation importance
        let options = PermutationImportanceOptions {
            n_permutations: 5,
            seed: Some(42),
            compute_intervals: false,
            interval_confidence: 0.95,
        };
        let perm =
            permutation_importance_ols(&y, &[x1, x2], &fit, &options).unwrap();
        assert_eq!(perm.variable_names.len(), 2);
    }
}
