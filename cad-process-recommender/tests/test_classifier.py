# tests/test_classifier.py
#
# Smoke tests and integration tests for ProcessClassifier.
# These tests use a lightweight synthetic model so they run fast in CI
# without needing the full pickled artifact from training.

import os
import pickle
import tempfile
import numpy as np
import pytest

from sklearn.preprocessing import SimpleImputer
import xgboost as xgb

from src.models.classifier import ProcessClassifier, PROCESS_CLASSES
from src.features.definitions import FEATURE_COLUMNS


def _build_toy_model():
    """Fits a minimal XGBoost model on random data — not accurate, just runnable."""
    np.random.seed(0)
    n, n_feat, n_cls = 200, len(FEATURE_COLUMNS), len(PROCESS_CLASSES)
    X = np.random.rand(n, n_feat)
    y = np.random.randint(0, n_cls, size=n)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        objective="multi:softprob",
        num_class=n_cls,
        verbosity=0,
        random_state=0,
    )
    model.fit(X_imp, y)
    return model, imputer


@pytest.fixture(scope="module")
def toy_classifier():
    model, imputer = _build_toy_model()
    clf = ProcessClassifier(xgb_model=model, imputer=imputer, calibrated=False)
    return clf


@pytest.fixture
def typical_features():
    """Typical CNC milling part feature vector."""
    return {
        "bounding_box_x": 85.0, "bounding_box_y": 60.0, "bounding_box_z": 20.0,
        "volume": 54200.0, "surface_area": 22800.0,
        "wall_thickness_min": 3.2, "wall_thickness_avg": 3.8,
        "aspect_ratio": 4.25, "hole_count": 3.0, "hole_diameter_min": 5.0,
        "undercut_flag": 0.0, "thin_wall_flag": 0.0,
        "curvature_complexity": 0.08, "symmetry_score": 0.32,
    }


class TestProcessClassifierInterface:
    def test_predict_proba_returns_array(self, toy_classifier, typical_features):
        proba = toy_classifier.predict_proba(typical_features)
        assert isinstance(proba, np.ndarray)

    def test_predict_proba_length_equals_n_classes(self, toy_classifier, typical_features):
        proba = toy_classifier.predict_proba(typical_features)
        assert len(proba) == len(PROCESS_CLASSES)

    def test_probabilities_sum_to_approximately_one(self, toy_classifier, typical_features):
        proba = toy_classifier.predict_proba(typical_features)
        assert abs(proba.sum() - 1.0) < 0.01

    def test_all_probabilities_nonnegative(self, toy_classifier, typical_features):
        proba = toy_classifier.predict_proba(typical_features)
        assert (proba >= 0).all()

    def test_predict_top_k_returns_k_items(self, toy_classifier, typical_features):
        results = toy_classifier.predict_top_k(typical_features, k=3)
        assert len(results) == 3

    def test_predict_top_1_returns_1_item(self, toy_classifier, typical_features):
        results = toy_classifier.predict_top_k(typical_features, k=1)
        assert len(results) == 1

    def test_top_k_results_are_sorted_descending(self, toy_classifier, typical_features):
        results = toy_classifier.predict_top_k(typical_features, k=5)
        probs = [r[2] for r in results]
        assert probs == sorted(probs, reverse=True)

    def test_top_k_tuples_have_three_elements(self, toy_classifier, typical_features):
        results = toy_classifier.predict_top_k(typical_features, k=3)
        for item in results:
            assert len(item) == 3  # (code, name, probability)

    def test_top_k_process_codes_are_in_class_list(self, toy_classifier, typical_features):
        results = toy_classifier.predict_top_k(typical_features, k=3)
        for code, name, prob in results:
            assert code in PROCESS_CLASSES

    def test_missing_optional_feature_handled(self, toy_classifier, typical_features):
        """hole_diameter_min is None when part has no holes — should not crash."""
        features = dict(typical_features)
        features["hole_diameter_min"] = None
        features["hole_count"] = 0.0
        # Should not raise
        proba = toy_classifier.predict_proba(features)
        assert len(proba) == len(PROCESS_CLASSES)


class TestSaveLoad:
    def test_save_and_reload(self, toy_classifier, typical_features, tmp_path):
        model_path = str(tmp_path / "test_model.pkl")
        toy_classifier.save(model_path)

        loaded = ProcessClassifier.load(model_path)
        original_proba = toy_classifier.predict_proba(typical_features)
        loaded_proba = loaded.predict_proba(typical_features)
        np.testing.assert_array_almost_equal(original_proba, loaded_proba, decimal=6)

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            ProcessClassifier.load("nonexistent/path/model.pkl")


class TestPropertyAccessors:
    def test_n_classes(self, toy_classifier):
        assert toy_classifier.n_classes == len(PROCESS_CLASSES)

    def test_feature_names(self, toy_classifier):
        assert toy_classifier.feature_names == list(FEATURE_COLUMNS)
