"""Tests for model save/load serialization."""

import pytest
import tempfile
import os
import json
import linreg_core


# Shared fixtures
Y = [3.1, 4.9, 7.2, 8.8, 11.1, 12.9, 15.0]
X = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]


class TestSaveModel:
    """Tests for save_model()."""

    def test_save_ols_returns_metadata(self, tmp_path):
        path = str(tmp_path / "ols.json")
        result = linreg_core.ols_regression(Y, X, ["Intercept", "X1"])
        meta = linreg_core.save_model(result, path)
        assert isinstance(meta, dict)
        assert meta["model_type"] == "OLS"
        assert meta["path"] == path
        assert "format_version" in meta
        assert "library_version" in meta

    def test_save_creates_file(self, tmp_path):
        path = str(tmp_path / "model.json")
        result = linreg_core.ols_regression(Y, X, ["Intercept", "X1"])
        linreg_core.save_model(result, path)
        assert os.path.exists(path), "save_model should create the file"

    def test_save_file_is_valid_json(self, tmp_path):
        path = str(tmp_path / "model.json")
        result = linreg_core.ols_regression(Y, X, ["Intercept", "X1"])
        linreg_core.save_model(result, path)
        with open(path) as f:
            data = json.load(f)
        assert "metadata" in data
        assert "data" in data

    def test_save_with_name(self, tmp_path):
        path = str(tmp_path / "named.json")
        result = linreg_core.ols_regression(Y, X, ["Intercept", "X1"])
        meta = linreg_core.save_model(result, path, name="My OLS Model")
        assert meta.get("name") == "My OLS Model"

    def test_save_ridge(self, tmp_path):
        path = str(tmp_path / "ridge.json")
        result = linreg_core.ridge_regression(Y, X, lambda_val=0.1)
        meta = linreg_core.save_model(result, path)
        assert meta["model_type"] == "Ridge"

    def test_save_lasso(self, tmp_path):
        path = str(tmp_path / "lasso.json")
        result = linreg_core.lasso_regression(Y, X, lambda_val=0.01)
        meta = linreg_core.save_model(result, path)
        assert meta["model_type"] == "Lasso"

    def test_save_elastic_net(self, tmp_path):
        path = str(tmp_path / "enet.json")
        result = linreg_core.elastic_net_regression(Y, X, lambda_val=0.01, alpha=0.5)
        meta = linreg_core.save_model(result, path)
        assert meta["model_type"] == "ElasticNet"

    def test_save_wls(self, tmp_path):
        path = str(tmp_path / "wls.json")
        weights = [1.0] * len(Y)
        result = linreg_core.wls_regression(Y, X, weights)
        meta = linreg_core.save_model(result, path)
        assert meta["model_type"] == "WLS"

    def test_save_invalid_type_raises(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with pytest.raises(Exception):
            linreg_core.save_model("not a result object", path)


class TestLoadModel:
    """Tests for load_model()."""

    def test_load_ols_roundtrip(self, tmp_path):
        path = str(tmp_path / "ols.json")
        original = linreg_core.ols_regression(Y, X, ["Intercept", "X1"])
        linreg_core.save_model(original, path)

        loaded = linreg_core.load_model(path)
        assert hasattr(loaded, "coefficients"), "Loaded model should have coefficients"
        assert len(loaded.coefficients) == len(original.coefficients)
        for orig, load in zip(original.coefficients, loaded.coefficients):
            assert abs(orig - load) < 1e-10, "Coefficients should match after roundtrip"

    def test_load_ridge_roundtrip(self, tmp_path):
        path = str(tmp_path / "ridge.json")
        original = linreg_core.ridge_regression(Y, X, lambda_val=0.1)
        linreg_core.save_model(original, path)
        loaded = linreg_core.load_model(path)
        assert hasattr(loaded, "coefficients")
        for orig, load in zip(original.coefficients, loaded.coefficients):
            assert abs(orig - load) < 1e-10

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(Exception):
            linreg_core.load_model("/nonexistent/path/model.json")

    def test_load_invalid_json_raises(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("not json at all")
        with pytest.raises(Exception):
            linreg_core.load_model(path)

    def test_load_missing_metadata_raises(self, tmp_path):
        path = str(tmp_path / "no_meta.json")
        with open(path, "w") as f:
            json.dump({"model": {"coefficients": [1.0, 2.0]}}, f)
        with pytest.raises(Exception):
            linreg_core.load_model(path)

    def test_save_load_preserves_r_squared(self, tmp_path):
        path = str(tmp_path / "ols.json")
        original = linreg_core.ols_regression(Y, X, ["Intercept", "X1"])
        linreg_core.save_model(original, path)
        loaded = linreg_core.load_model(path)
        assert abs(original.r_squared - loaded.r_squared) < 1e-10
