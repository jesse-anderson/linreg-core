// ============================================================================
// Serialization Unit Tests
// ============================================================================
//
// Comprehensive tests for model serialization (save/load) functionality.
// Tests cover round-trip serialization for all model types, error handling,
// version compatibility, and metadata preservation.

use linreg_core::core::ols_regression;
use linreg_core::loess::{loess_fit, LoessOptions};
use linreg_core::regularized::{elastic_net_fit, lasso_fit, ridge_fit, ElasticNetOptions, LassoFitOptions, RidgeFitOptions};
use linreg_core::serialization::{ModelLoad, ModelSave, FORMAT_VERSION};
use linreg_core::serialization::types::{ModelMetadata, ModelType, SerializedModel};
use linreg_core::weighted_regression::wls_regression;
use linreg_core::Error;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

// ============================================================================
// Test Constants and Helpers
// ============================================================================

const EPSILON: f64 = 1e-10;
const STAT_TOLERANCE: f64 = 1e-6;

/// Helper function to assert two f64 values are close within tolerance
fn assert_close(a: f64, b: f64, tolerance: f64, context: &str) {
    let diff = (a - b).abs();
    assert!(
        diff <= tolerance,
        "{}: {} != {}, diff = {} (tolerance = {})",
        context, a, b, diff, tolerance
    );
}

/// Helper function to assert vectors are approximately equal
fn assert_vec_close(a: &[f64], b: &[f64], tolerance: f64, context: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{}: Length mismatch {} vs {}",
        context, a.len(), b.len()
    );
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        assert_close(av, bv, tolerance, &format!("{}[{}]", context, i));
    }
}

/// Creates a temporary file path for testing
fn temp_file_path(name: &str) -> String {
    // Use a simple temp file approach without tempfile crate
    let temp_dir = std::env::temp_dir();
    let mut path = PathBuf::from(temp_dir);
    // Use a unique name based on the test name and a random component
    let random_suffix: u64 = unsafe { std::arch::x86_64::_rdtsc() };
    path.push(format!("{}_{}.json", name, random_suffix));
    path.to_str().unwrap().to_string()
}

/// Cleans up a test file
fn cleanup_file(path: &str) {
    let _ = fs::remove_file(path);
}

// ============================================================================
// OLS Round-Trip Tests
// ============================================================================

#[test]
fn test_ols_round_trip() {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let names = vec!["Intercept".to_string(), "X".to_string()];

    let original = ols_regression(&y, &[x.clone()], &names).expect("OLS should succeed");

    let path = temp_file_path("ols_model.json");

    // Save
    original
        .save(&path)
        .expect("Save should succeed");

    // Load
    let loaded = linreg_core::core::RegressionOutput::load(&path)
        .expect("Load should succeed");

    // Compare all fields
    assert_vec_close(&original.coefficients, &loaded.coefficients, EPSILON, "coefficients");
    assert_vec_close(&original.std_errors, &loaded.std_errors, EPSILON, "std_errors");
    assert_vec_close(&original.t_stats, &loaded.t_stats, EPSILON, "t_stats");
    assert_vec_close(&original.p_values, &loaded.p_values, EPSILON, "p_values");
    assert_close(original.r_squared, loaded.r_squared, EPSILON, "r_squared");
    assert_close(original.adj_r_squared, loaded.adj_r_squared, EPSILON, "adj_r_squared");
    assert_close(original.f_statistic, loaded.f_statistic, EPSILON, "f_statistic");
    assert_close(original.f_p_value, loaded.f_p_value, EPSILON, "f_p_value");
    assert_eq!(original.n, loaded.n);
    assert_eq!(original.k, loaded.k);

    cleanup_file(&path);
}

#[test]
fn test_ols_round_trip_with_custom_name() {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let names = vec!["Intercept".to_string(), "X".to_string()];

    let original = ols_regression(&y, &[x], &names).expect("OLS should succeed");

    let path = temp_file_path("ols_model_named.json");
    let custom_name = "My Custom OLS Model".to_string();

    // Save with custom name
    original
        .save_with_name(&path, Some(custom_name.clone()))
        .expect("Save should succeed");

    // Load
    let loaded = linreg_core::core::RegressionOutput::load(&path)
        .expect("Load should succeed");

    // Verify the model loaded correctly
    assert_eq!(original.coefficients.len(), loaded.coefficients.len());

    // Also verify the file contains the custom name by reading it directly
    let content = fs::read_to_string(&path).expect("Should read file");
    assert!(content.contains("My Custom OLS Model"), "File should contain custom name");

    cleanup_file(&path);
}

// ============================================================================
// Ridge Round-Trip Tests
// ============================================================================

#[test]
fn test_ridge_round_trip() {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    use linreg_core::linalg::Matrix;
    let mut data = vec![1.0; y.len() * 2];
    for (i, &val) in x.iter().enumerate() {
        data[i * 2 + 1] = val;
    }
    let x_matrix = Matrix::new(y.len(), 2, data);

    let options = RidgeFitOptions {
        lambda: 1.0,
        standardize: false,
        intercept: true,
        ..Default::default()
    };

    let original = ridge_fit(&x_matrix, &y, &options).expect("Ridge should succeed");

    let path = temp_file_path("ridge_model.json");

    // Save
    original
        .save(&path)
        .expect("Save should succeed");

    // Load
    let loaded = linreg_core::regularized::ridge::RidgeFit::load(&path)
        .expect("Load should succeed");

    // Compare all fields
    assert_close(original.intercept, loaded.intercept, STAT_TOLERANCE, "intercept");
    assert_vec_close(&original.coefficients, &loaded.coefficients, STAT_TOLERANCE, "coefficients");
    assert_close(original.lambda, loaded.lambda, EPSILON, "lambda");
    assert_close(original.r_squared, loaded.r_squared, STAT_TOLERANCE, "r_squared");
    assert_close(original.adj_r_squared, loaded.adj_r_squared, STAT_TOLERANCE, "adj_r_squared");
    assert_vec_close(&original.fitted_values, &loaded.fitted_values, STAT_TOLERANCE, "fitted_values");
    assert_vec_close(&original.residuals, &loaded.residuals, STAT_TOLERANCE, "residuals");

    cleanup_file(&path);
}

// ============================================================================
// Lasso Round-Trip Tests
// ============================================================================

#[test]
fn test_lasso_round_trip() {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    use linreg_core::linalg::Matrix;
    let mut data = vec![1.0; y.len() * 2];
    for (i, &val) in x.iter().enumerate() {
        data[i * 2 + 1] = val;
    }
    let x_matrix = Matrix::new(y.len(), 2, data);

    let options = LassoFitOptions {
        lambda: 0.1,
        standardize: false,
        intercept: true,
        ..Default::default()
    };

    let original = lasso_fit(&x_matrix, &y, &options).expect("Lasso should succeed");

    let path = temp_file_path("lasso_model.json");

    // Save
    original
        .save(&path)
        .expect("Save should succeed");

    // Load
    let loaded = linreg_core::regularized::lasso::LassoFit::load(&path)
        .expect("Load should succeed");

    // Compare all fields
    assert_close(original.intercept, loaded.intercept, STAT_TOLERANCE, "intercept");
    assert_vec_close(&original.coefficients, &loaded.coefficients, STAT_TOLERANCE, "coefficients");
    assert_close(original.lambda, loaded.lambda, EPSILON, "lambda");
    assert_eq!(original.n_nonzero, loaded.n_nonzero);
    assert_eq!(original.converged, loaded.converged);

    cleanup_file(&path);
}

// ============================================================================
// Elastic Net Round-Trip Tests
// ============================================================================

#[test]
fn test_elastic_net_round_trip() {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    use linreg_core::linalg::Matrix;
    let mut data = vec![1.0; y.len() * 2];
    for (i, &val) in x.iter().enumerate() {
        data[i * 2 + 1] = val;
    }
    let x_matrix = Matrix::new(y.len(), 2, data);

    let options = ElasticNetOptions {
        lambda: 0.1,
        alpha: 0.5,
        standardize: false,
        intercept: true,
        ..Default::default()
    };

    let original = elastic_net_fit(&x_matrix, &y, &options).expect("Elastic Net should succeed");

    let path = temp_file_path("elastic_net_model.json");

    // Save
    original
        .save(&path)
        .expect("Save should succeed");

    // Load
    let loaded = linreg_core::regularized::elastic_net::ElasticNetFit::load(&path)
        .expect("Load should succeed");

    // Compare all fields
    assert_close(original.intercept, loaded.intercept, STAT_TOLERANCE, "intercept");
    assert_vec_close(&original.coefficients, &loaded.coefficients, STAT_TOLERANCE, "coefficients");
    assert_close(original.lambda, loaded.lambda, EPSILON, "lambda");
    assert_close(original.alpha, loaded.alpha, EPSILON, "alpha");
    assert_eq!(original.n_nonzero, loaded.n_nonzero);
    assert_eq!(original.converged, loaded.converged);

    cleanup_file(&path);
}

// ============================================================================
// WLS Round-Trip Tests
// ============================================================================

#[test]
fn test_wls_round_trip() {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];

    let original = wls_regression(&y, &[x.clone()], &weights).expect("WLS should succeed");

    let path = temp_file_path("wls_model.json");

    // Save
    original
        .save(&path)
        .expect("Save should succeed");

    // Load
    let loaded = linreg_core::weighted_regression::wls::WlsFit::load(&path)
        .expect("Load should succeed");

    // Compare all fields
    assert_vec_close(&original.coefficients, &loaded.coefficients, STAT_TOLERANCE, "coefficients");
    assert_vec_close(&original.standard_errors, &loaded.standard_errors, STAT_TOLERANCE, "standard_errors");
    assert_close(original.r_squared, loaded.r_squared, STAT_TOLERANCE, "r_squared");
    assert_eq!(original.n, loaded.n);
    assert_eq!(original.k, loaded.k);

    cleanup_file(&path);
}

// ============================================================================
// LOESS Round-Trip Tests
// ============================================================================

#[test]
fn test_loess_round_trip() {
    let y = vec![1.0, 2.5, 2.0, 4.5, 5.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let options = LoessOptions::default();

    let original = loess_fit(&y, &[x.clone()], &options).expect("LOESS should succeed");

    let path = temp_file_path("loess_model.json");

    // Save
    original
        .save(&path)
        .expect("Save should succeed");

    // Load
    let loaded = linreg_core::loess::types::LoessFit::load(&path)
        .expect("Load should succeed");

    // Compare all fields
    assert_vec_close(&original.fitted, &loaded.fitted, STAT_TOLERANCE, "fitted");
    assert_close(original.span, loaded.span, EPSILON, "span");
    assert_eq!(original.degree, loaded.degree);
    assert_eq!(original.robust_iterations, loaded.robust_iterations);

    cleanup_file(&path);
}

// ============================================================================
// File I/O Error Tests
// ============================================================================

#[test]
fn test_load_nonexistent_file() {
    let result = linreg_core::core::RegressionOutput::load("/nonexistent/path/model.json");

    match result {
        Err(Error::IoError(_)) => (),
        _ => panic!("Expected IoError for nonexistent file, got {:?}", result),
    }
}

#[test]
fn test_load_malformed_json() {
    let path = temp_file_path("malformed.json");

    // Write invalid JSON
    let mut file = fs::File::create(&path).unwrap();
    file.write_all(b"{ this is not valid json }").unwrap();
    file.sync_all().unwrap();

    let result = linreg_core::core::RegressionOutput::load(&path);

    match result {
        Err(Error::DeserializationError(_)) => (),
        _ => panic!("Expected DeserializationError for malformed JSON, got {:?}", result),
    }

    cleanup_file(&path);
}

#[test]
fn test_save_to_invalid_path() {
    let y = vec![2.0, 4.0, 6.0];
    let x = vec![1.0, 2.0, 3.0];
    let names = vec!["Intercept".to_string(), "X".to_string()];

    let model = ols_regression(&y, &[x], &names).expect("OLS should succeed");

    // Try to save to a path with nonexistent directories
    let result = model.save("/nonexistent/directory/path/model.json");

    match result {
        Err(Error::IoError(_)) => (),
        _ => panic!("Expected IoError for invalid path, got {:?}", result),
    }
}

// ============================================================================
// Type Mismatch Tests
// ============================================================================

#[test]
fn test_type_mismatch_ols_as_ridge() {
    let y = vec![2.0, 4.0, 6.0];
    let x = vec![1.0, 2.0, 3.0];
    let names = vec!["Intercept".to_string(), "X".to_string()];

    let model = ols_regression(&y, &[x], &names).expect("OLS should succeed");

    let path = temp_file_path("ols_model.json");
    model.save(&path).expect("Save should succeed");

    // Try to load OLS model as Ridge
    let result = linreg_core::regularized::ridge::RidgeFit::load(&path);

    match result {
        Err(Error::ModelTypeMismatch { expected, found }) => {
            assert_eq!(expected, "Ridge");
            assert_eq!(found, "OLS");
        }
        _ => panic!("Expected ModelTypeMismatch error, got {:?}", result),
    }

    cleanup_file(&path);
}

#[test]
fn test_type_mismatch_loess_as_lasso() {
    let y = vec![1.0, 2.0, 3.0];
    let x = vec![1.0, 2.0, 3.0];

    let options = LoessOptions::default();
    let model = loess_fit(&y, &[x], &options).expect("LOESS should succeed");

    let path = temp_file_path("loess_model.json");
    model.save(&path).expect("Save should succeed");

    // Try to load LOESS model as Lasso
    let result = linreg_core::regularized::lasso::LassoFit::load(&path);

    match result {
        Err(Error::ModelTypeMismatch { expected, found }) => {
            assert_eq!(expected, "Lasso");
            assert_eq!(found, "LOESS");
        }
        _ => panic!("Expected ModelTypeMismatch error, got {:?}", result),
    }

    cleanup_file(&path);
}

// ============================================================================
// JSON Format Validation Tests
// ============================================================================

#[test]
fn test_json_format_structure() {
    let y = vec![2.0, 4.0, 6.0];
    let x = vec![1.0, 2.0, 3.0];
    let names = vec!["Intercept".to_string(), "X".to_string()];

    let model = ols_regression(&y, &[x], &names).expect("OLS should succeed");

    let path = temp_file_path("format_test.json");
    model.save(&path).expect("Save should succeed");

    // Read and parse the JSON
    let content = fs::read_to_string(&path).expect("Should read file");
    let parsed: serde_json::Value = serde_json::from_str(&content).expect("Should parse JSON");

    // Verify structure
    assert!(parsed.is_object(), "Root should be an object");
    assert!(parsed.get("metadata").is_some(), "Should have metadata field");
    assert!(parsed.get("data").is_some(), "Should have data field");

    // Verify metadata fields
    let metadata = parsed.get("metadata").unwrap();
    assert!(metadata.get("format_version").is_some(), "Should have format_version");
    assert!(metadata.get("library_version").is_some(), "Should have library_version");
    assert!(metadata.get("model_type").is_some(), "Should have model_type");
    assert!(metadata.get("created_at").is_some(), "Should have created_at");

    // Verify model type is correct
    let model_type = metadata.get("model_type").unwrap().as_str().unwrap();
    assert_eq!(model_type, "OLS");

    // Verify data contains expected fields
    let data = parsed.get("data").unwrap();
    assert!(data.get("coefficients").is_some(), "Should have coefficients");
    assert!(data.get("r_squared").is_some(), "Should have r_squared");

    cleanup_file(&path);
}

#[test]
fn test_format_version_constant() {
    // Verify the format version is set
    assert!(!FORMAT_VERSION.is_empty(), "FORMAT_VERSION should not be empty");
    assert_eq!(FORMAT_VERSION, "1.0", "Initial format version should be 1.0");
}

// ============================================================================
// Metadata Tests
// ============================================================================

#[test]
fn test_metadata_creation() {
    let metadata = ModelMetadata::new(ModelType::OLS, "1.0.0".to_string());

    assert_eq!(metadata.model_type, ModelType::OLS);
    assert_eq!(metadata.library_version, "1.0.0");
    assert_eq!(metadata.format_version, "1.0");
    assert!(metadata.created_at.len() > 0, "Should have timestamp");
    assert!(metadata.name.is_none(), "Name should be None by default");
}

#[test]
fn test_metadata_with_name() {
    let metadata = ModelMetadata::new(ModelType::Ridge, "1.0.0".to_string())
        .with_name("Test Model".to_string());

    assert_eq!(metadata.name.as_ref().unwrap(), "Test Model");
}

#[test]
fn test_model_type_display() {
    assert_eq!(ModelType::OLS.to_string(), "OLS");
    assert_eq!(ModelType::Ridge.to_string(), "Ridge");
    assert_eq!(ModelType::Lasso.to_string(), "Lasso");
    assert_eq!(ModelType::ElasticNet.to_string(), "ElasticNet");
    assert_eq!(ModelType::WLS.to_string(), "WLS");
    assert_eq!(ModelType::LOESS.to_string(), "LOESS");
}

#[test]
fn test_model_type_from_str() {
    assert_eq!("OLS".parse::<ModelType>().unwrap(), ModelType::OLS);
    assert_eq!("Ridge".parse::<ModelType>().unwrap(), ModelType::Ridge);
    assert_eq!("Lasso".parse::<ModelType>().unwrap(), ModelType::Lasso);
    assert_eq!("ElasticNet".parse::<ModelType>().unwrap(), ModelType::ElasticNet);
    assert_eq!("WLS".parse::<ModelType>().unwrap(), ModelType::WLS);
    assert_eq!("LOESS".parse::<ModelType>().unwrap(), ModelType::LOESS);

    // Test invalid type
    assert!("InvalidType".parse::<ModelType>().is_err());
}

// ============================================================================
// Cross-Model Type Tests
// ============================================================================

#[test]
fn test_all_model_types_have_save_load() {
    // This test verifies that all model types implement ModelSave and ModelLoad
    // by actually using the save and load methods

    let y = vec![1.0, 2.0, 3.0];
    let x = vec![1.0, 2.0, 3.0];

    // OLS
    {
        let names = vec!["Intercept".to_string(), "X".to_string()];
        let model = ols_regression(&y, &[x.clone()], &names).unwrap();
        let path = temp_file_path("test_ols.json");
        model.save(&path).expect("OLS save should work");
        let _loaded = linreg_core::core::RegressionOutput::load(&path)
            .expect("OLS load should work");
        cleanup_file(&path);
    }

    // Ridge
    {
        use linreg_core::linalg::Matrix;
        let x_matrix = Matrix::new(3, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
        let options = RidgeFitOptions::default();
        let model = ridge_fit(&x_matrix, &y, &options).unwrap();
        let path = temp_file_path("test_ridge.json");
        model.save(&path).expect("Ridge save should work");
        let _loaded = linreg_core::regularized::ridge::RidgeFit::load(&path)
            .expect("Ridge load should work");
        cleanup_file(&path);
    }

    // Lasso
    {
        use linreg_core::linalg::Matrix;
        let x_matrix = Matrix::new(3, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
        let options = LassoFitOptions::default();
        let model = lasso_fit(&x_matrix, &y, &options).unwrap();
        let path = temp_file_path("test_lasso.json");
        model.save(&path).expect("Lasso save should work");
        let _loaded = linreg_core::regularized::lasso::LassoFit::load(&path)
            .expect("Lasso load should work");
        cleanup_file(&path);
    }

    // Elastic Net
    {
        use linreg_core::linalg::Matrix;
        let x_matrix = Matrix::new(3, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
        let options = ElasticNetOptions::default();
        let model = elastic_net_fit(&x_matrix, &y, &options).unwrap();
        let path = temp_file_path("test_enet.json");
        model.save(&path).expect("Elastic Net save should work");
        let _loaded = linreg_core::regularized::elastic_net::ElasticNetFit::load(&path)
            .expect("Elastic Net load should work");
        cleanup_file(&path);
    }

    // WLS
    {
        let weights = vec![1.0, 1.0, 1.0];
        let model = wls_regression(&y, &[vec![1.0, 2.0, 3.0]], &weights).unwrap();
        let path = temp_file_path("test_wls.json");
        model.save(&path).expect("WLS save should work");
        let _loaded = linreg_core::weighted_regression::wls::WlsFit::load(&path)
            .expect("WLS load should work");
        cleanup_file(&path);
    }

    // LOESS
    {
        let options = LoessOptions::default();
        let model = loess_fit(&y, &[vec![1.0, 2.0, 3.0]], &options).unwrap();
        let path = temp_file_path("test_loess.json");
        model.save(&path).expect("LOESS save should work");
        let _loaded = linreg_core::loess::types::LoessFit::load(&path)
            .expect("LOESS load should work");
        cleanup_file(&path);
    }
}
