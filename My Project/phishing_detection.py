# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load the Dataset
try:
    df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")  # Ensure the file is in the same folder
    print("\n✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("\n❌ Error: Dataset file not found! Check the file path.")
    exit()

# Step 2: Preprocess the Data
df.columns = df.columns.str.strip().str.lower()  # Normalize column names

# Identify Label Column
label_col = None
for col in df.columns:
    if "label" in col or "status" in col or "phishing" in col:
        label_col = col
        break

if label_col is None:
    print("\n❌ Error: Label column not found! Check dataset column names.")
    print("\nAvailable Columns:", df.columns)
    exit()

print(f"\n✅ Using '{label_col}' as the label column.")

# Step 3: Convert All Columns to Numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Step 4: Handle Missing Values
df.fillna(df.mean(), inplace=True)

# Step 5: Prepare Data for Training
X = df.drop([label_col], axis=1)
y = df[label_col]

# Step 6: Verify Data Before Splitting
if len(X) == 0 or len(y) == 0:
    print("\n❌ Error: Dataset is empty after cleaning! Check your data preprocessing.")
    exit()

print("\n✅ Data is ready for training!")

# Step 7: Split Data into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n✅ Data split completed! Training samples:", len(X_train), "Testing samples:", len(X_test))

# Step 8: Train the Machine Learning Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\n🎯 Model training complete!")

# Step 9: Save the Model
joblib.dump(model, "phishing_detector.pkl")
print("\n✅ Model saved as phishing_detector.pkl")

# Step 10: Save Test Data for Testing
test_data = {"X_test": X_test, "y_test": y_test}
joblib.dump(test_data, "test_data.pkl")

print("\n✅ Test dataset saved as test_data.pkl")