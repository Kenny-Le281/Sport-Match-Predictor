# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load the dataset
# Replace the path with the downloaded CSV file's location
data = pd.read_csv('sports_data_mock.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Step 3: Preprocess the data
# Encode categorical features (HomeTeam, AwayTeam, VenueType) using one-hot encoding
data = pd.get_dummies(data, columns=["HomeTeam", "AwayTeam", "VenueType"], drop_first=True)

# Map the target column (GameResult) to binary values (Win = 1, Loss = 0)
data['GameResult'] = data['GameResult'].map({'Win': 1, 'Loss': 0})

# Step 4: Define features (X) and target (y)
X = data.drop('GameResult', axis=1)
y = data['GameResult']

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 9: Feature importance analysis
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()

# Step 10: Save the trained model (optional)
import joblib
joblib.dump(model, 'sports_winner_prediction_model.pkl')