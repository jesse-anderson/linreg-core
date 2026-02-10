//! CSV parsing for WASM
//!
//! Provides CSV parsing functionality that can be called from JavaScript.

#![cfg(feature = "wasm")]

use std::collections::HashSet;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use super::domain::check_domain;
use crate::error::{error_json, error_to_json};

#[derive(Serialize)]
struct ParsedCsv {
    headers: Vec<String>,
    data: Vec<serde_json::Map<String, serde_json::Value>>,
    numeric_columns: Vec<String>,
}

/// Parses CSV data and returns it as a JSON string.
///
/// Parses the CSV content and identifies numeric columns. Returns a JSON object
/// with headers, data rows, and a list of numeric column names.
///
/// # Arguments
///
/// * `content` - CSV content as a string
///
/// # Returns
///
/// JSON string with structure:
/// ```json
/// {
///   "headers": ["col1", "col2", ...],
///   "data": [{"col1": 1.0, "col2": "text"}, ...],
///   "numeric_columns": ["col1", ...]
/// }
/// ```
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn parse_csv(content: &str) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(content.as_bytes());

    // Get headers
    let headers: Vec<String> = match reader.headers() {
        Ok(h) => h.iter().map(|s| s.to_string()).collect(),
        Err(e) => return error_json(&format!("Failed to read headers: {}", e)),
    };

    let mut data = Vec::new();
    let mut numeric_col_set = HashSet::new();

    for result in reader.records() {
        let record = match result {
            Ok(r) => r,
            Err(e) => return error_json(&format!("Failed to parse CSV record: {}", e)),
        };

        if record.len() != headers.len() {
            continue;
        }

        let mut row_map = serde_json::Map::new();

        for (i, field) in record.iter().enumerate() {
            if i >= headers.len() {
                continue;
            }

            let header = &headers[i];
            let val_trimmed = field.trim();

            // Try to parse as f64
            if let Ok(num) = val_trimmed.parse::<f64>() {
                if num.is_finite() {
                    row_map.insert(
                        header.clone(),
                        serde_json::Value::Number(serde_json::Number::from_f64(num).unwrap()),
                    );
                    numeric_col_set.insert(header.clone());
                    continue;
                }
            }

            // Fallback to string
            row_map.insert(
                header.clone(),
                serde_json::Value::String(val_trimmed.to_string()),
            );
        }
        data.push(row_map);
    }

    let mut numeric_columns: Vec<String> = numeric_col_set.into_iter().collect();
    numeric_columns.sort();

    let output = ParsedCsv {
        headers,
        data,
        numeric_columns,
    };

    serde_json::to_string(&output).unwrap_or_else(|_| error_json("Failed to serialize CSV output"))
}
