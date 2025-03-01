import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the trained model
try:
    model = joblib.load("phishing_detector.pkl")
    print("\n✅ Model loaded successfully!")
except FileNotFoundError:
    print("\n❌ Error: Model file not found! Train the model first.")
    exit()

# Load Test Data
try:
    test_data = joblib.load("test_data.pkl")
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    print("\n✅ Test data loaded successfully!")
except FileNotFoundError:
    print("\n❌ Error: Test dataset not found! Run 'phishing_detection.py' first.")
    exit()

# Step 1: Predict on Test Data
y_pred = model.predict(X_test)

# Step 2: Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Testing Accuracy: {accuracy * 100:.2f}%")

# Step 3: Get the Feature Names from X_test
feature_names = list(X_test.columns)  # Get feature names from the dataset

# Step 4: Create a New Example with the Correct Number of Features and Column Names
new_website_data = np.zeros((1, len(feature_names)))  # Create a zero array
new_website_df = pd.DataFrame(new_website_data, columns=feature_names)  # Convert to DataFrame

# Replace the first 5 values with actual feature data
new_website_df.iloc[0, :5] = [1, 30, 5, 2, 0]  # Modify with actual values

# Step 5: Predict on the New Example
prediction = model.predict(new_website_df)

if prediction[0] == 1:
    print("🚨 Phishing site detected!")
else:
    print("✅ This site is safe.")