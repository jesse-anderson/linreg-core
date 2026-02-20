//! Unit tests for feature importance module.

mod feature_importance {
    use linreg_core::feature_importance::{
        permutation_importance_ols, shap_values_linear, standardized_coefficients, vif_ranking,
        PermutationImportanceOptions,
    };
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
