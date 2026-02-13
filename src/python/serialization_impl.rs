// ============================================================================
// Model Serialization (Save/Load) for Python
// ============================================================================

#[cfg(feature = "python")]
use pyo3::types::PyDict;
use crate::serialization::types::{ModelType, SerializedModel};

// ============================================================================
// Save Model
// ============================================================================

/// Save a trained model to a file.
///
/// Args:
///     result: A regression result object (OLSResult, RidgeResult, LassoResult,
///             ElasticNetResult, LoessResult, or WlsResult)
///     path: File path to save the model (will be created/overwritten)
///     name: Optional custom name for the model
///
/// Returns:
///     A dictionary with metadata about the saved model
///
/// Example:
///     >>> result = ols_regression(y, [x1, x2], ["Intercept", "X1", "X2"])
///     >>> metadata = save_model(result, "my_model.json", name="My Housing Model")
///     >>> print(metadata["model_type"])
///     OLS
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (result, path, name=None))]
fn save_model(
    py: Python,
    result: &Bound<PyAny>,
    path: String,
    name: Option<String>,
) -> PyResult<PyObject> {
    // Get the result type as a string
    let type_obj = result.get_type();
    let type_name: String = type_obj.name()?.to_string();

    // Try to get __dict__ from the result object
    let dict_attr = result.getattr("__dict__")
        .map_err(|_| pyo3::PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Result object must have a __dict__ attribute"
        ))?;
    let result_dict = dict_attr.downcast::<PyDict>()
        .map_err(|_| pyo3::PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected __dict__ to be a dict"
        ))?;

    // Determine model type
    let model_type = match type_name.as_str() {
        "OLSResult" => ModelType::OLS,
        "RidgeResult" => ModelType::Ridge,
        "LassoResult" => ModelType::Lasso,
        "ElasticNetResult" => ModelType::ElasticNet,
        "LoessResult" => ModelType::LOESS,
        "WlsResult" => ModelType::WLS,
        _ => {
            return Err(pyo3::PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                format!("Unknown result type: {}", type_name)
            ));
        }
    };

    // Convert Python dict to serde_json::Value
    let model_data = pydict_to_json(py, result_dict)?;

    // Create metadata
    let mut metadata = crate::serialization::types::ModelMetadata::new(
        model_type,
        env!("CARGO_PKG_VERSION").to_string()
    );
    if let Some(n) = name {
        metadata = metadata.with_name(n);
    }

    // Create serialized model and save
    let serialized = SerializedModel::new(metadata, model_data);
    crate::serialization::json::save_to_file(&serialized, &path)
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyIOError, _>(
            format!("Failed to save model: {}", e)
        ))?;

    // Return metadata as dict
    let result_dict = PyDict::new_bound(py);
    result_dict.set_item("model_type", serialized.metadata.model_type.to_string())?;
    result_dict.set_item("path", path)?;
    result_dict.set_item("format_version", crate::serialization::FORMAT_VERSION)?;
    result_dict.set_item("library_version", env!("CARGO_PKG_VERSION"))?;
    if let Some(n) = &serialized.metadata.name {
        result_dict.set_item("name", n)?;
    }

    Ok(result_dict.into())
}

// ============================================================================
// Load Model
// ============================================================================

/// Load a trained model from a file.
///
/// Args:
///     path: File path to load the model from
///
/// Returns:
///     The appropriate result object (OLSResult, RidgeResult, etc.)
///
/// Example:
///     >>> result = load_model("my_model.json")
///     >>> print(result.coefficients)
#[cfg(feature = "python")]
#[pyfunction]
fn load_model(py: Python, path: String) -> PyResult<PyObject> {
    // Load the serialized model
    let serialized = crate::serialization::json::load_from_file(&path)
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyIOError, _>(
            format!("Failed to load model from '{}': {}", path, e)
        ))?;

    // Convert the model data to a Python dict
    let py_dict_bound = json_to_pydict(py, &serialized.data)?;

    // Get the result class from the module
    let module = py.import_bound("linreg_core")?;

    // Get the class based on model type
    let class_name = match serialized.metadata.model_type {
        ModelType::OLS => "OLSResult",
        ModelType::Ridge => "RidgeResult",
        ModelType::Lasso => "LassoResult",
        ModelType::ElasticNet => "ElasticNetResult",
        ModelType::LOESS => "LoessResult",
        ModelType::WLS => "WlsResult",
    };

    let result_class = module.getattr(class_name)?;

    // Create the result object from the dict
    let result = result_class.call((), Some(&py_dict_bound))?;

    Ok(result.into())
}

// ============================================================================
// Helper: Convert PyDict to serde_json::Value
// ============================================================================

#[cfg(feature = "python")]
fn pydict_to_json(py: Python, dict: &Bound<PyDict>) -> PyResult<serde_json::Value> {
    use serde_json::Value;

    let mut map = serde_json::map::Map::new();
    for (key, val) in dict.iter() {
        let key_str: String = key.extract()?;
        let json_val = pyany_to_json(py, val)?;
        map.insert(key_str, json_val);
    }

    Ok(Value::Object(map))
}

#[cfg(feature = "python")]
fn pyany_to_json(py: Python, value: Bound<PyAny>) -> PyResult<serde_json::Value> {
    use serde_json::Value;
    use pyo3::types::PyList;

    // Try None first
    if value.is_none() {
        return Ok(Value::Null);
    }

    // Try bool
    if let Ok(b) = value.extract::<bool>() {
        return Ok(Value::Bool(b));
    }

    // Try integers
    if let Ok(i) = value.extract::<i64>() {
        return Ok(Value::Number(i.into()));
    }

    // Try floats
    if let Ok(f) = value.extract::<f64>() {
        return Ok(Value::Number(serde_json::Number::from_f64(f).unwrap_or_else(|| serde_json::Number::from(0))));
    }

    // Try string
    if let Ok(s) = value.extract::<String>() {
        return Ok(Value::String(s));
    }

    // Try list/tuple
    if let Ok(list) = value.downcast::<PyList>() {
        let mut arr = Vec::new();
        for item in list.iter() {
            arr.push(pyany_to_json(py, item)?);
        }
        return Ok(Value::Array(arr));
    }

    // Try dict
    if let Ok(d) = value.downcast::<PyDict>() {
        return pydict_to_json(py, d);
    }

    // Fallback: convert to string
    let s: String = value.str()?.extract()?;
    Ok(Value::String(s))
}

// ============================================================================
// Helper: Convert serde_json::Value to Python dict
// ============================================================================

#[cfg(feature = "python")]
fn json_to_pydict<'a>(py: Python<'a>, value: &serde_json::Value) -> PyResult<Bound<'a, PyDict>> {
    let dict = PyDict::new_bound(py);

    if let serde_json::Value::Object(map) = value {
        for (key, val) in map {
            let py_val = json_to_py(py, val)?;
            dict.set_item(key, py_val)?;
        }
    }

    Ok(dict)
}

#[cfg(feature = "python")]
fn json_to_py(py: Python, value: &serde_json::Value) -> PyResult<PyObject> {
    use pyo3::types::PyList;
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.into_py(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_py(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_py(py))
            } else {
                Ok(py.None())
            }
        },
        serde_json::Value::String(s) => Ok(s.into_py(py)),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into())
        },
        serde_json::Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (key, val) in map {
                let py_val = json_to_py(py, val)?;
                dict.set_item(key, py_val)?;
            }
            Ok(dict.into())
        },
    }
}
