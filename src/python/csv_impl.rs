// ============================================================================
// CSV Parsing (Native Types API)
// ============================================================================


/// Parse CSV content and return a CSVResult object.
///
/// Args:
///     content: CSV content as a string
///
/// Returns:
///     CSVResult object with headers, data, numeric_columns, n_rows, n_cols
#[cfg(feature = "python")]
#[pyfunction]
fn parse_csv(content: &str) -> PyResult<crate::python::PyCSVResult> {
    use pyo3::types::PyDict;
    use pyo3::types::PyList;

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(content.as_bytes());

    let headers: Vec<String> = reader.headers()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CSV error: {}", e)))?
        .iter().map(|s| s.to_string()).collect();

    let n_cols = headers.len();
    let mut numeric_col_set = std::collections::HashSet::new();

    // Build Python list of dicts directly
    Python::with_gil(|py| {
        let data_list = PyList::empty_bound(py);

        for result in reader.records() {
            let record = result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CSV error: {}", e)))?;
            if record.len() != headers.len() {
                continue;
            }

            let row_dict = PyDict::new_bound(py);

            for (i, field) in record.iter().enumerate() {
                if i >= headers.len() {
                    continue;
                }

                let header = &headers[i];
                let val_trimmed = field.trim();

                if let Ok(num) = val_trimmed.parse::<f64>() {
                    if num.is_finite() {
                        row_dict.set_item(header, num)?;
                        numeric_col_set.insert(header.clone());
                        continue;
                    }
                }

                row_dict.set_item(header, val_trimmed)?;
            }
            data_list.append(row_dict)?;
        }

        let mut numeric_columns: Vec<String> = numeric_col_set.into_iter().collect();
        numeric_columns.sort();

        let n_rows = data_list.len();

        // Use struct literal construction (not .new()) to avoid PyO3 conflicts
        Ok(crate::python::PyCSVResult {
            headers,
            data: data_list.into(),
            numeric_columns,
            n_rows,
            n_cols,
        })
    })
}
