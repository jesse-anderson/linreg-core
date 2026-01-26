// ============================================================================
// Test Fixtures
// ============================================================================
//
// Shared test data and helper functions used across all WASM test modules.

#![cfg(target_arch = "wasm32")]

/// Returns housing price data as JSON string (25 observations)
pub fn get_housing_y() -> String {
    serde_json::to_string(&vec![
        245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1, 445.8, 167.9, 367.4,
        289.6, 198.2, 478.5, 256.3, 334.7, 178.5, 398.9, 223.4, 312.5, 156.8, 423.7, 267.9,
    ])
    .unwrap()
}

/// Returns housing predictor variables as JSON string
/// [square_feet, bedrooms, age] - each with 25 observations
pub fn get_housing_x_vars() -> String {
    let square_feet = vec![
        1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0, 2200.0,
        900.0, 1950.0, 1500.0, 1050.0, 2600.0, 1300.0, 1850.0, 1000.0, 2100.0, 1250.0, 1700.0,
        850.0, 2350.0, 1400.0,
    ];
    let bedrooms = vec![
        3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0, 4.0, 2.0, 4.0, 3.0, 2.0, 5.0, 3.0, 4.0,
        2.0, 4.0, 3.0, 3.0, 2.0, 4.0, 3.0,
    ];
    let age = vec![
        15.0, 10.0, 25.0, 5.0, 8.0, 12.0, 20.0, 2.0, 18.0, 7.0, 3.0, 30.0, 6.0, 14.0, 22.0, 1.0,
        16.0, 9.0, 28.0, 4.0, 19.0, 11.0, 35.0, 3.0, 13.0,
    ];
    serde_json::to_string(&vec![square_feet, bedrooms, age]).unwrap()
}

/// Returns variable names as JSON string
pub fn get_variable_names() -> String {
    serde_json::to_string(&vec!["Intercept", "Square_Feet", "Bedrooms", "Age"]).unwrap()
}
