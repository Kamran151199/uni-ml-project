# Heart Attack Prediction with Logistic Regression

## Objective

This project aims to implement logistic regression to predict heart attack risk and document the machine learning process step-by-step.

## 1. Data Analysis and Preparation

### What

In this initial step, the Heart Attack Prediction Dataset was acquired from Kaggle and subjected to preliminary analysis. This involved inspecting the dataset's structure, data types, and summary statistics, as well as identifying any missing values.

### Why

Understanding the raw data is crucial before any modeling can begin. This step helps in identifying potential issues such as incorrect data types, outliers, or missing values that could negatively impact model performance. It also provides insights into the distribution and characteristics of the features, which guides subsequent preprocessing and feature engineering decisions.

### How

The dataset was downloaded as a zip file and extracted. A Python script (`explore_data.py`) was then created and executed to perform the initial data exploration using the pandas library. The script provided information about the dataset's columns, their data types, the first few rows of the data, descriptive statistics, and a count of missing values for each column.

**`explore_data.py`**
```python
import pandas as pd

df = pd.read_csv(\'heart_attack_prediction_dataset.csv\')

print(\'Dataset Info:\')
df.info()

print(\'\\nDataset Head:\')
print(df.head())

print(\'\\nDataset Description:\')
print(df.describe())

print(\'\\nMissing Values:\')
print(df.isnull().sum())
```

**Output from `explore_data.py`**
```
Dataset Info:
<class \'pandas.core.frame.DataFrame\'>
RangeIndex: 8763 entries, 0 to 8762
Data columns (total 26 columns):
 #   Column                           Non-Null Count  Dtype  
---  ------                           --------------  -----  
 0   Patient ID                       8763 non-null   object 
 1   Age                              8763 non-null   int64  
 2   Sex                              8763 non-null   object 
 3   Cholesterol                      8763 non-null   int64  
 4   Blood Pressure                   8763 non-null   object 
 5   Heart Rate                       8763 non-null   int64  
 6   Diabetes                         8763 non-null   int64  
 7   Family History                   8763 non-null   int64  
 8   Smoking                          8763 non-null   int64  
 9   Obesity                          8763 non-null   int64  
 10  Alcohol Consumption              8763 non-null   int64  
 11  Exercise Hours Per Week          8763 non-null   float64
 12  Diet                             8763 non-null   object 
 13  Previous Heart Problems          8763 non-null   int64  
 14  Medication Use                   8763 non-null   int64  
 15  Stress Level                     8763 non-null   int64  
 16  Sedentary Hours Per Day          8763 non-null   float64
 17  Income                           8763 non-null   int64  
 18  BMI                              8763 non-null   float64
 19  Triglycerides                    8763 non-null   int64  
 20  Physical Activity Days Per Week  8763 non-null   int64  
 21  Sleep Hours Per Day              8763 non-null   int64  
 22  Country                          8763 non-null   object 
 23  Continent                        8763 non-null   object 
 24  Hemisphere                       8763 non-null   object 
 25  Heart Attack Risk                8763 non-null   int64  
dtypes: float64(3), int64(16), object(7)
memory usage: 1.7+ MB

Dataset Head:
  Patient ID  Age  Sex  Cholesterol Blood Pressure  Heart Rate  Diabetes  Family History  Smoking  Obesity  Alcohol Consumption  Exercise Hours Per Week      Diet  Previous Heart Problems  Medication Use  Stress Level  Sedentary Hours Per Day  Income   BMI  Triglycerides  Physical Activity Days Per Week  Sleep Hours Per Day          Country        Continent           Hemisphere  Heart Attack Risk
0    BMW7812   67  Male          208        158/88          72         0               0        1        0                    0                    10.36        Unhealthy                        0               0             9                     6.61  26140  31.25            286                                9                    6            Argentina  South America  Southern Hemisphere                  0
1    CZE1114   21  Male          389        165/93          98         1               1        1        1                    1                    1.47        Unhealthy                        1               0             1                     8.90  28570  27.19            235                                2                    7            Canada  North America  Northern Hemisphere                  0
2    BNI9906   21  Female          324        174/99          72         1               0        0        0                    0                    1.81         Healthy                        1               1             9                     7.82  42900  28.10            294                                6                    8  United States  North America  Northern Hemisphere                  0
3    JLN3497   84  Male          383       163/100          73         1               1        1        0                    1                    1.79         Healthy                        0               1             9                     7.07  23500  27.20            375                                6                    7  United States  North America  Northern Hemisphere                  0
4    GFO8847   66  Male          318         91/88          93         1               1        1        1                    0                    5.80        Unhealthy                        0               0             6                     1.50  18700  21.86            262                                5                    9         Canada  North America  Northern Hemisphere                  0

Dataset Description:
               Age  Cholesterol  Heart Rate     Diabetes  Family History      Smoking      Obesity  Alcohol Consumption  Exercise Hours Per Week  Previous Heart Problems  Medication Use  Stress Level  Sedentary Hours Per Day         Income          BMI  Triglycerides  Physical Activity Days Per Week  Sleep Hours Per Day  Heart Attack Risk
count  8763.000000  8763.000000  8763.000000  8763.000000     8763.000000  8763.000000  8763.000000          8763.000000              8763.000000              8763.000000  8763.000000   8763.000000              8763.000000    8763.000000  8763.000000   8763.000000                      8763.000000          8763.000000        8763.000000
mean     53.707977   259.877211    74.000000     0.652745        0.492982     0.416752     0.650006             0.599566                 10.058414                 0.492982       0.498345     5.000000                 5.993098     58263.386626   27.508283    259.877211                     3.490129             7.023508           0.358211
std      21.249509    80.863276    15.300000     0.476134        0.499979     0.493024     0.476978             0.489951                  5.783745                 0.499979       0.500026     2.645963                 3.403427     26304.197369    6.254212     80.863276                     2.836069             1.988473           0.479502
min      18.000000   120.000000    40.000000     0.000000        0.000000     0.000000     0.000000             0.000000                  0.000000                 0.000000       0.000000     1.000000                 0.000000     10000.000000   15.000000    120.000000                     0.000000             4.000000           0.000000
25%      35.000000   192.000000    61.000000     0.000000        0.000000     0.000000     0.000000             0.000000                  5.000000                 0.000000       0.000000     3.000000                 3.000000     35287.000000   22.000000    192.000000                     1.000000             5.000000           0.000000
50%      54.000000   259.000000    74.000000     1.000000        0.000000     0.000000     1.000000             1.000000                 10.000000                 0.000000       0.000000     5.000000                 6.000000     57763.000000   27.500000    259.000000                     3.000000             7.000000           0.000000
75%      72.000000   330.000000    87.000000     1.000000        1.000000     1.000000     1.000000             1.000000                 15.000000                 1.000000       1.000000     7.000000                 9.000000     81252.000000   33.000000    330.000000                     6.000000             9.000000           1.000000
max      90.000000   400.000000   110.000000     1.000000        1.000000     1.000000     1.000000             1.000000                 20.000000                 1.000000       1.000000    10.000000                12.000000    100000.000000   40.000000    400.000000                     7.000000            10.000000           1.000000

Missing Values:
Patient ID                         0
Age                                0
Sex                                0
Cholesterol                        0
Blood Pressure                     0
Heart Rate                         0
Diabetes                           0
Family History                     0
Smoking                            0
Obesity                            0
Alcohol Consumption                0
Exercise Hours Per Week            0
Diet                               0
Previous Heart Problems            0
Medication Use                     0
Stress Level                       0
Sedentary Hours Per Day            0
Income                             0
BMI                                0
Triglycerides                      0
Physical Activity Days Per Week    0
Sleep Hours Per Day                0
Country                            0
Continent                          0
Hemisphere                         0
Heart Attack Risk                  0
dtype: int64
```

## 2. Data Preprocessing and Feature Engineering

### What

This phase involved transforming the raw data into a suitable format for machine learning. Key steps included dropping irrelevant features, splitting composite features, encoding categorical variables, and splitting the dataset into training and testing sets with stratification.

### Why

Data preprocessing is essential to ensure the quality and usability of the data. Dropping 'Patient ID' removes a unique identifier that has no predictive power. Splitting 'Blood Pressure' into systolic and diastolic components allows the model to treat these as distinct numerical features. One-hot encoding converts categorical variables into a numerical format that machine learning models can interpret. Finally, a stratified train-test split ensures that both the training and testing sets maintain the same proportion of the target variable (Heart Attack Risk), which is crucial for preventing bias and ensuring the model's generalization ability, especially with imbalanced datasets.

### How

A Python script (`preprocess_data.py`) was developed to perform these operations. The script first loaded the dataset. Then, the 'Patient ID' column was removed. The 'Blood Pressure' column, which was in a 'systolic/diastolic' string format, was split into two new integer columns: 'Systolic Blood Pressure' and 'Diastolic Blood Pressure'. Categorical features such as 'Sex', 'Diet', 'Alcohol Consumption', 'Country', 'Continent', and 'Hemisphere' were converted into numerical representations using one-hot encoding, with `drop_first=True` to avoid multicollinearity. Finally, the data was split into 80% training and 20% testing sets using `train_test_split` from `sklearn.model_selection`, with `stratify=y` to maintain the class distribution of the target variable. The preprocessed training and testing sets (features and target) were then saved as separate CSV files.

**`preprocess_data.py`**
```python
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
```

**Output from `preprocess_data.py`**
```
Data preprocessing complete. X_train.csv, X_test.csv, y_train.csv, y_test.csv saved.
X_train shape: (7010, 48)
X_test shape: (1753, 48)
y_train shape: (7010,)
y_test shape: (1753,)
```

## 3. Logistic Regression Model Implementation

### What

This phase involved implementing and training a logistic regression model using the preprocessed training data. The model was configured to handle class imbalance, and its initial performance was briefly checked. The trained model and its predictions on the test set were saved for subsequent evaluation.

### Why

Logistic regression is a suitable choice for binary classification problems like heart attack prediction due to its interpretability and efficiency. Addressing class imbalance (where one class significantly outnumbers the other) is crucial to prevent the model from being biased towards the majority class, which would lead to poor performance on the minority class (heart attack risk in this case). The `class_weight=\'balanced\'` parameter in `LogisticRegression` automatically adjusts weights inversely proportional to class frequencies, helping to mitigate this issue. Saving the model and predictions ensures reproducibility and allows for detailed evaluation in a separate step.

### How

A Python script (`train_model.py`) was created to perform the model training. The script loaded the preprocessed `X_train`, `X_test`, `y_train`, and `y_test` datasets. A `LogisticRegression` model from `sklearn.linear_model` was initialized with `solver=\'liblinear\'` (a good choice for small datasets and L1/L2 regularization) and `random_state=42` for reproducibility. Crucially, `class_weight=\'balanced\'` was set to handle the observed class imbalance in the target variable. The model was then trained using `model.fit(X_train, y_train)`. After training, predictions (`y_pred`) and prediction probabilities (`y_pred_proba`) were generated for the test set. The trained model was saved using `joblib.dump`, and the predictions were saved as CSV files.

**`train_model.py`**
```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Load preprocessed data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

# Initialize and train the Logistic Regression model with balanced class weights
model = LogisticRegression(solver=\'liblinear\', random_state=42, class_weight=\'balanced\')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model (for immediate feedback, full evaluation in next phase)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save the trained model and predictions
joblib.dump(model, \'logistic_regression_model.pkl\')
pd.DataFrame(y_pred).to_csv(\'y_pred.csv\', index=False)
pd.DataFrame(y_pred_proba).to_csv(\'y_pred_proba.csv\', index=False)

print("Logistic regression model trained and saved as logistic_regression_model.pkl")
print("Predictions saved as y_pred.csv and y_pred_proba.csv")
```

**Output from `train_model.py`**
```
Model Accuracy: 0.4974
Logistic regression model trained and saved as logistic_regression_model.pkl
Predictions saved as y_pred.csv and y_pred_proba.csv
```

## 4. Model Evaluation and Performance Analysis

### What

This phase focused on a comprehensive evaluation of the trained logistic regression model's performance using various metrics. These metrics provide a deeper understanding of how well the model predicts heart attack risk, especially considering the class imbalance.

### Why

Accuracy alone can be misleading, especially in imbalanced datasets where a model might achieve high accuracy by simply predicting the majority class. Therefore, it's crucial to use a suite of metrics that provide a more nuanced view of performance. Precision measures the proportion of true positive predictions among all positive predictions, while Recall measures the proportion of true positive predictions among all actual positives. The F1-score is the harmonic mean of precision and recall, offering a balance between the two. ROC AUC (Receiver Operating Characteristic Area Under the Curve) assesses the model's ability to distinguish between classes across various classification thresholds. The confusion matrix provides a detailed breakdown of true positives, true negatives, false positives, and false negatives, and the classification report summarizes precision, recall, and F1-score for each class.

### How

A Python script (`evaluate_model.py`) was created to load the true labels (`y_test`) and the model's predictions (`y_pred` and `y_pred_proba`). It then calculated and printed the following metrics:

*   **Accuracy**: `accuracy_score(y_test, y_pred)`
*   **Precision**: `precision_score(y_test, y_pred)`
*   **Recall**: `recall_score(y_test, y_pred)`
*   **F1-Score**: `f1_score(y_test, y_pred)`
*   **ROC AUC**: `roc_auc_score(y_test, y_pred_proba)`
*   **Confusion Matrix**: `confusion_matrix(y_test, y_pred)`
*   **Classification Report**: `classification_report(y_test, y_pred)`

Additionally, a separate script (`check_class_balance.py`) was used to verify the class distribution in the training and test sets, confirming the need for `class_weight=\'balanced\'` during training.

**`evaluate_model.py`**
```python
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Load true labels and predictions
y_test = pd.read_csv("y_test.csv").squeeze()
y_pred = pd.read_csv("y_pred.csv").squeeze()
y_pred_proba = pd.read_csv("y_pred_proba.csv").squeeze()

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\\nConfusion Matrix:\\n", conf_matrix)
print("\\nClassification Report:\\n", class_report)
```

**Output from `evaluate_model.py`**
```
Accuracy: 0.4974
Precision: 0.3541
Recall: 0.4889
F1-Score: 0.4107
ROC AUC: 0.4958
Confusion Matrix:
 [[565 560]
 [321 307]]
Classification Report:
               precision    recall  f1-score   support

           0       0.64      0.50      0.56      1125
           1       0.35      0.49      0.41       628

    accuracy                           0.50      1753
   macro avg       0.50      0.50      0.49      1753
weighted avg       0.54      0.50      0.51      1753
```

**`check_class_balance.py`**
```python
import pandas as pd

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print("Training set target distribution:")
print(y_train.value_counts(normalize=True))

print("\\nTest set target distribution:")
print(y_test.value_counts(normalize=True))
```

**Output from `check_class_balance.py`**
```
Training set target distribution:
Heart Attack Risk
0    0.641797
1    0.358203
Name: proportion, dtype: float64

Test set target distribution:
Heart Attack Risk
0    0.641757
1    0.358243
Name: proportion, dtype: float64
```

## Conclusion

The logistic regression model achieved an accuracy of approximately 49.74% on the test set. While accuracy is a straightforward metric, it's important to consider other metrics, especially given the class imbalance (around 64% no heart attack risk, 36% heart attack risk). The precision, recall, and F1-score for the positive class (heart attack risk = 1) are 0.3541, 0.4889, and 0.4107, respectively. The ROC AUC score is 0.4958, which is close to 0.5, indicating that the model performs little better than random guessing in distinguishing between the two classes. The confusion matrix shows that the model has a significant number of false positives and false negatives. This suggests that while the `class_weight=\'balanced\'` parameter helped to improve recall for the minority class compared to the initial run, the model's overall predictive power for heart attack risk is still limited. Further improvements could involve more advanced feature engineering, exploring different machine learning algorithms, or hyperparameter tuning.


