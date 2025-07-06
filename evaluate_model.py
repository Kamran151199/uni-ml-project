import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Load true labels and predictions
y_test = pd.read_csv("y_test.csv").squeeze()
y_pred = pd.read_csv("y_pred.csv").squeeze()
y_pred_proba = pd.read_csv("y_pred_proba.csv").squeeze()

# Calculate evaluation metrics
# Accuracy: Percentage of correct predictions out of total predictions
# Shows overall model performance but can be misleading with imbalanced datasets
accuracy = accuracy_score(y_test, y_pred)

# Precision: Of all positive predictions, how many were actually correct (TP / (TP + FP))
# Important for heart attack prediction - reduces false alarms that could cause unnecessary anxiety
precision = precision_score(y_test, y_pred)

# Recall (Sensitivity): Of all actual positive cases, how many were correctly identified (TP / (TP + FN))
# Critical for heart attack prediction - we want to catch as many actual cases as possible
recall = recall_score(y_test, y_pred)

# F1-Score: Harmonic mean of precision and recall (2 * precision * recall / (precision + recall))
# Balances precision and recall, useful when you need both to be reasonably high
f1 = f1_score(y_test, y_pred)

# ROC AUC: Area Under the Receiver Operating Characteristic Curve
# Measures the model's ability to distinguish between classes across all thresholds
# Values closer to 1.0 indicate better discrimination ability
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Confusion Matrix: Shows true positives, false positives, true negatives, false negatives
# Provides detailed breakdown of prediction accuracy for each class
conf_matrix = confusion_matrix(y_test, y_pred)

# Classification Report: Comprehensive summary including precision, recall, F1-score for each class
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
