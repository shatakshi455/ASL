import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

def load_dataset():
    """Loads dataset from pickle file."""
    with open('../datasets/datasetMN.pickle', 'rb') as f:
        data_dict = pickle.load(f)
    data = np.asarray(data_dict['data'])  # Shape: (samples, features)
    labels = np.asarray(data_dict['labels'])  # Shape: (samples,)
    
    return data, labels

def train_model(data, labels):
    """Trains and evaluates the model."""
    # Split data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
    )

    # Train a RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=200,  
        max_depth=30,  
        min_samples_split=5,
        random_state=42
    )
    model.fit(x_train, y_train)

    # Predict and evaluate
    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f'{accuracy * 100:.2f}% of samples were classified correctly!')

    # Cross-validation score
    cv_scores = cross_val_score(model, x_train, y_train, cv=5)
    print(f'Cross-validation accuracy: {cv_scores.mean() * 100:.2f}%')

    return model

def save_model(model, filename="../models/model_scalerMN.p"):
    """Saves the trained model."""
    with open(filename, 'wb') as f:
        pickle.dump({'model': model}, f)

# Ensure script runs only when executed directly
if __name__ == "__main__":
    data, labels = load_dataset()
    
    model = train_model(data, labels)
    save_model(model)
