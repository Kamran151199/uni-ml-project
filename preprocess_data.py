
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("heart_attack_prediction_dataset.csv")

# Drop Patient ID
df = df.drop("Patient ID", axis=1)

# Split Blood Pressure into Systolic and Diastolic
df[["Systolic Blood Pressure", "Diastolic Blood Pressure"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
df = df.drop("Blood Pressure", axis=1)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=["Sex", "Diet", "Alcohol Consumption", "Country", "Continent", "Hemisphere"], drop_first=True)

# Define features (X) and target (y)
X = df.drop("Heart Attack Risk", axis=1)
y = df["Heart Attack Risk"]

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save preprocessed data
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Data preprocessing complete. X_train.csv, X_test.csv, y_train.csv, y_test.csv saved.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


