import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Load dataset
with open('./dataset1.pickle', 'rb') as f:
    data_dict = pickle.load(f)

print(type(data_dict['data']))
x = 0
for i in data_dict['data']:
    x =  x + 1
    if(len(i) != 67): print(len(i))
print("x ",x)
 
data = np.asarray(data_dict['data'])  # Shape: (samples, features)
labels = np.asarray(data_dict['labels'])  # Shape: (samples,)

# Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# # Normalize the features (important for 3D coordinates)
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

 
# Train a tuned RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=200,  # More trees for better accuracy
    max_depth=30,  # Prevent overfitting
    min_samples_split=5,
    random_state=42
)

model.fit(x_train, y_train)

# Predict and evaluate
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)

print(f'{accuracy * 100:.2f}% of samples were classified correctly!')

# Cross-validation score
cv_scores = cross_val_score(model, x_train, y_train, cv=5)
print(f'Cross-validation accuracy: {cv_scores.mean() * 100:.2f}%')

# Save model and scaler
with open('model_scaler3.p', 'wb') as f:
    pickle.dump({'model': model}, f)