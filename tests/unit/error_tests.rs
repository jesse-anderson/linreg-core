// ============================================================================
// Error Handling Unit Tests
// ============================================================================
//
// Tests for all error variants, display formatting, and JSON serialization.

use linreg_core::{Error, error_json, error_to_json};

// ============================================================================
// Error Display Tests
// ============================================================================

#[test]
fn test_error_display_singular_matrix() {
    let err = Error::SingularMatrix;
    let display = format!("{}", err);

    assert!(
        display.contains("singular") || display.contains("Singular"),
        "SingularMatrix display should mention 'singular': {}",
        display
    );
    assert!(
        display.contains("multicollinearity"),
        "SingularMatrix display should mention 'multicollinearity': {}",
        display
    );
}

#[test]
fn test_error_display_insufficient_data() {
    let err = Error::InsufficientData { required: 10, available: 5 };
    let display = format!("{}", err);

    assert!(
        display.contains("Insufficient") || display.contains("insufficient"),
        "InsufficientData display should mention 'Insufficient': {}",
        display
    );
    assert!(
        display.contains("10"),
        "InsufficientData display should mention required value: {}",
        display
    );
    assert!(
        display.contains("5"),
        "InsufficientData display should mention available value: {}",
        display
    );
}

#[test]
fn test_error_display_invalid_input() {
    let err = Error::InvalidInput("Test error message".to_string());
    let display = format!("{}", err);

    assert!(
        display.contains("Invalid input") || display.contains("Invalid"),
        "InvalidInput display should mention 'Invalid': {}",
        display
    );
    assert!(
        display.contains("Test error message"),
        "InvalidInput display should contain the message: {}",
        display
    );
}

#[test]
fn test_error_display_parse_error() {
    let err = Error::ParseError("Failed to parse CSV".to_string());
    let display = format!("{}", err);

    assert!(
        display.contains("Parse") || display.contains("parse"),
        "ParseError display should mention 'Parse': {}",
        display
    );
    assert!(
        display.contains("Failed to parse CSV"),
        "ParseError display should contain the message: {}",
        display
    );
}

#[test]
fn test_error_display_domain_check() {
    let err = Error::DomainCheck("Unauthorized domain: example.com".to_string());
    let display = format!("{}", err);

    assert!(
        display.contains("Domain") || display.contains("domain"),
        "DomainCheck display should mention 'Domain': {}",
        display
    );
    assert!(
        display.contains("Unauthorized domain"),
        "DomainCheck display should contain the message: {}",
        display
    );
}

// ============================================================================
// Error JSON Format Tests
// ============================================================================

#[test]
fn test_error_json_format() {
    let json = error_json("Test error");

    // Should be valid JSON with "error" field
    assert!(json.contains("\"error\""), "JSON should have 'error' field");
    assert!(json.contains("Test error"), "JSON should contain the message");
}

#[test]
fn test_error_json_exact_format() {
    let json = error_json("Test message");

    // Parse as JSON to verify structure
    let parsed: serde_json::Value = serde_json::from_str(&json)
        .expect("Should be valid JSON");

    assert!(parsed.is_object(), "Should be a JSON object");
    assert!(parsed.get("error").is_some(), "Should have 'error' field");
    assert_eq!(
        parsed.get("error").unwrap().as_str(),
        Some("Test message"),
        "Error message should match"
    );
}

#[test]
fn test_error_to_json_singular_matrix() {
    let err = Error::SingularMatrix;
    let json = error_to_json(&err);

    let parsed: serde_json::Value = serde_json::from_str(&json)
        .expect("Should be valid JSON");

    assert!(parsed.get("error").is_some());
    let error_msg = parsed.get("error").unwrap().as_str().unwrap();
    assert!(error_msg.contains("singular") || error_msg.contains("Singular"));
}

#[test]
fn test_error_to_json_insufficient_data() {
    let err = Error::InsufficientData { required: 10, available: 5 };
    let json = error_to_json(&err);

    let parsed: serde_json::Value = serde_json::from_str(&json)
        .expect("Should be valid JSON");

    assert!(parsed.get("error").is_some());
    let error_msg = parsed.get("error").unwrap().as_str().unwrap();
    assert!(error_msg.contains("10") || error_msg.contains("5"));
}

#[test]
fn test_error_to_json_invalid_input() {
    let err = Error::InvalidInput("Bad parameter".to_string());
    let json = error_to_json(&err);

    let parsed: serde_json::Value = serde_json::from_str(&json)
        .expect("Should be valid JSON");

    assert!(parsed.get("error").is_some());
    let error_msg = parsed.get("error").unwrap().as_str().unwrap();
    assert!(error_msg.contains("Bad parameter"));
}

#[test]
fn test_error_to_json_parse_error() {
    let err = Error::ParseError("JSON parse failed".to_string());
    let json = error_to_json(&err);

    let parsed: serde_json::Value = serde_json::from_str(&json)
        .expect("Should be valid JSON");

    assert!(parsed.get("error").is_some());
    let error_msg = parsed.get("error").unwrap().as_str().unwrap();
    assert!(error_msg.contains("JSON parse failed"));
}

#[test]
fn test_error_to_json_domain_check() {
    let err = Error::DomainCheck("Unauthorized".to_string());
    let json = error_to_json(&err);

    let parsed: serde_json::Value = serde_json::from_str(&json)
        .expect("Should be valid JSON");

    assert!(parsed.get("error").is_some());
    let error_msg = parsed.get("error").unwrap().as_str().unwrap();
    assert!(error_msg.contains("Unauthorized"));
}

// ============================================================================
// Error Equality Tests
// ============================================================================

#[test]
fn test_error_equality_singular_matrix() {
    let err1 = Error::SingularMatrix;
    let err2 = Error::SingularMatrix;

    assert_eq!(err1, err2, "SingularMatrix errors should be equal");
}

#[test]
fn test_error_equality_insufficient_data() {
    let err1 = Error::InsufficientData { required: 10, available: 5 };
    let err2 = Error::InsufficientData { required: 10, available: 5 };

    assert_eq!(err1, err2, "InsufficientData errors should be equal");
}

#[test]
fn test_error_inequality_insufficient_data() {
    let err1 = Error::InsufficientData { required: 10, available: 5 };
    let err2 = Error::InsufficientData { required: 8, available: 5 };

    assert_ne!(err1, err2, "InsufficientData errors with different values should not be equal");
}

#[test]
fn test_error_equality_invalid_input() {
    let err1 = Error::InvalidInput("msg".to_string());
    let err2 = Error::InvalidInput("msg".to_string());

    assert_eq!(err1, err2, "InvalidInput errors with same message should be equal");
}

#[test]
fn test_error_inequality_different_variants() {
    let errors = vec![
        Error::SingularMatrix,
        Error::InsufficientData { required: 10, available: 5 },
        Error::InvalidInput("test".to_string()),
        Error::ParseError("parse".to_string()),
        Error::DomainCheck("domain".to_string()),
    ];

    for i in 0..errors.len() {
        for j in (i + 1)..errors.len() {
            assert_ne!(
                errors[i], errors[j],
                "Error variant {} should not equal variant {}",
                i, j
            );
        }
    }
}

// ============================================================================
// All Error Variants Display Test
// ============================================================================

#[test]
fn test_all_error_variants_display() {
    let errors = vec![
        Error::SingularMatrix,
        Error::InsufficientData { required: 10, available: 5 },
        Error::InvalidInput("test message".to_string()),
        Error::ParseError("parse error".to_string()),
        Error::DomainCheck("domain error".to_string()),
    ];

    for err in errors {
        let display = format!("{}", err);
        assert!(
            !display.is_empty(),
            "Error display should not be empty: {:?}",
            err
        );
        assert!(
            display.len() > 10,
            "Error display should be meaningful: {:?} -> {}",
            err, display
        );
    }
}

#[test]
fn test_all_error_variants_json_serialization() {
    let errors = vec![
        Error::SingularMatrix,
        Error::InsufficientData { required: 10, available: 5 },
        Error::InvalidInput("test message".to_string()),
        Error::ParseError("parse error".to_string()),
        Error::DomainCheck("domain error".to_string()),
    ];

    for err in errors {
        let json = error_to_json(&err);

        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json)
            .expect(&format!("Should be valid JSON for error: {:?}", err));

        // Should have error field
        assert!(
            parsed.get("error").is_some(),
            "JSON should have 'error' field for error: {:?}",
            err
        );

        // Error message should not be empty
        let error_msg = parsed.get("error").unwrap().as_str().unwrap();
        assert!(
            !error_msg.is_empty(),
            "Error message should not be empty for error: {:?}",
            err
        );
    }
}
