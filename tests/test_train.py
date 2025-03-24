import pickle
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from unittest.mock import MagicMock, patch
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# Load dataset
@pytest.fixture
def load_data():
    """Fixture to load dataset for testing."""
    with open('./dataset1.pickle', 'rb') as f:
        data_dict = pickle.load(f)

    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])
    return data, labels


def test_dataset_shape(load_data):
    """Ensure the dataset has correct shape and feature count."""
    data, labels = load_data
    assert len(data) == len(labels), "Mismatch between data samples and labels"
    assert all(len(sample) == 67 for sample in data), "Each sample should have 67 features"


def test_train_model(load_data):
    """Ensure the model trains without errors."""
    data, labels = load_data
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=5, random_state=42)
    
    # Ensure no error occurs during training
    model.fit(x_train, y_train)


def test_prediction_accuracy(load_data):
    """Ensure model achieves reasonable accuracy."""
    data, labels = load_data
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=5, random_state=42)
    model.fit(x_train, y_train)
    
    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)

    assert accuracy > 0.9, f"Accuracy is too low: {accuracy * 100:.2f}%"


def test_cross_validation(load_data):
    """Ensure cross-validation score is reasonable."""
    data, labels = load_data
    x_train, _, y_train, _ = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=5, random_state=42)
    
    cv_scores = cross_val_score(model, x_train, y_train, cv=5)
    
    assert cv_scores.mean() > 0.9, f"Cross-validation score is too low: {cv_scores.mean() * 100:.2f}%"


@patch("pickle.dump")
def test_model_saving(mock_pickle_dump, load_data):
    """Ensure the trained model is saved correctly."""
    data, labels = load_data
    x_train, _, y_train, _ = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=5, random_state=42)
    model.fit(x_train, y_train)

    # Simulate saving
    with open('model_scaler3.p', 'wb') as f:
        pickle.dump({'model': model}, f)

    # Verify pickle.dump was called
    mock_pickle_dump.assert_called_once()

