import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import time
import os


def explore_data():
    """Step 1: Explore the dataset"""
    print("=" * 70)
    print("STEP 1: DATA EXPLORATION")
    print("=" * 70)
    
    df = pd.read_csv('heart_attack_prediction_dataset.csv')
    
    print('Dataset Info:')
    df.info()
    
    print('\nDataset Head:')
    print(df.head())
    
    print('\nDataset Description:')
    print(df.describe())
    
    print('\nMissing Values:')
    print(df.isnull().sum())
    
    print('\nTarget Variable Distribution:')
    print(df['Heart Attack Risk'].value_counts(normalize=True))
    
    return df


def preprocess_data():
    """Step 2: Preprocess the data"""
    print("\n" + "=" * 70)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 70)
    
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
    
    print("Data preprocessing complete. Files saved:")
    print("- X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def check_class_balance():
    """Step 3: Check class balance in train/test sets"""
    print("\n" + "=" * 70)
    print("STEP 3: CLASS BALANCE ANALYSIS")
    print("=" * 70)
    
    y_train = pd.read_csv("y_train.csv").squeeze()
    y_test = pd.read_csv("y_test.csv").squeeze()
    
    print("Training set target distribution:")
    print(y_train.value_counts(normalize=True))
    
    print("\nTest set target distribution:")
    print(y_test.value_counts(normalize=True))
    
    # Check if classes are balanced
    train_balance = y_train.value_counts(normalize=True)
    test_balance = y_test.value_counts(normalize=True)
    
    print(f"\nClass balance analysis:")
    print(f"Training set - Class 0: {train_balance[0]:.3f}, Class 1: {train_balance[1]:.3f}")
    print(f"Test set - Class 0: {test_balance[0]:.3f}, Class 1: {test_balance[1]:.3f}")
    
    if abs(train_balance[0] - train_balance[1]) > 0.1:
        print("⚠️  Classes are imbalanced - using class_weight='balanced' in model training")
    else:
        print("✅ Classes are reasonably balanced")


def train_model():
    """Step 4: Train logistic regression model with different solvers"""
    print("\n" + "=" * 70)
    print("STEP 4: MODEL TRAINING")
    print("=" * 70)
    
    # Load preprocessed data
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").squeeze()
    y_test = pd.read_csv("y_test.csv").squeeze()
    
    # Dictionary of different gradient descent solvers to try
    solvers = {
        'lbfgs': 'Limited-memory BFGS (quasi-Newton method)',
        'sag': 'Stochastic Average Gradient',
        'saga': 'Stochastic Average Gradient with averaging',
        'newton-cholesky': 'Newton-CG (conjugate gradient)',
        'liblinear': 'Coordinate descent (current method)'
    }
    
    print("Comparing different gradient descent solvers for Logistic Regression:")
    print("-" * 70)
    
    best_model = None
    best_accuracy = 0
    best_solver = None
    
    for solver_name, description in solvers.items():
        print(f"\nTesting {solver_name}: {description}")
        
        try:
            # Start timing
            start_time = time.time()
            
            # Initialize model with current solver
            model = LogisticRegression(
                solver=solver_name, 
                random_state=42, 
                class_weight='balanced',
                max_iter=1000  # Increased iterations for gradient descent convergence
            )
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Training time
            training_time = time.time() - start_time
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")
            print(f"  Training time: {training_time:.2f} seconds")
            print(f"  Converged: {model.n_iter_ if hasattr(model, 'n_iter_') else 'N/A'} iterations")
            
            # Keep track of best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_solver = solver_name
                best_predictions = y_pred
                best_probabilities = y_pred_proba
                
        except Exception as e:
            print(f"  Error with {solver_name}: {str(e)}")
    
    print("\n" + "-" * 70)
    print(f"Best performing solver: {best_solver} with accuracy: {best_accuracy:.4f}")
    
    # Save the best model and predictions
    joblib.dump(best_model, 'logistic_regression_model.pkl')
    pd.DataFrame(best_predictions).to_csv('y_pred.csv', index=False)
    pd.DataFrame(best_probabilities).to_csv('y_pred_proba.csv', index=False)
    
    print(f"\nBest model ({best_solver}) saved as logistic_regression_model.pkl")
    print("Predictions saved as y_pred.csv and y_pred_proba.csv")
    
    return best_model, best_solver


def evaluate_model():
    """Step 5: Evaluate the trained model"""
    print("\n" + "=" * 70)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 70)
    
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
    
    print("FINAL MODEL PERFORMANCE:")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    
    # Interpretation of metrics for heart attack prediction
    print("\n" + "-" * 70)
    print("METRIC INTERPRETATION FOR HEART ATTACK PREDICTION:")
    print("-" * 70)
    print("• Accuracy: Overall correctness of predictions")
    print("• Precision: Of predicted heart attacks, how many were correct (reduces false alarms)")
    print("• Recall: Of actual heart attacks, how many were caught (critical for patient safety)")
    print("• F1-Score: Balance between precision and recall")
    print("• ROC AUC: Model's ability to distinguish between risk levels")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


def main():
    """Complete Heart Attack Prediction Pipeline"""
    print("🫀 HEART ATTACK PREDICTION - COMPLETE ML PIPELINE")
    print("=" * 70)
    
    try:
        # Step 1: Data Exploration
        df = explore_data()
        
        # Step 2: Data Preprocessing
        X_train, X_test, y_train, y_test = preprocess_data()
        
        # Step 3: Class Balance Check
        check_class_balance()
        
        # Step 4: Model Training
        best_model, best_solver = train_model()
        
        # Step 5: Model Evaluation
        metrics = evaluate_model()
        
        # Pipeline Summary
        print("\n" + "=" * 70)
        print("🎯 PIPELINE EXECUTION SUMMARY")
        print("=" * 70)
        print("✅ Data exploration completed")
        print("✅ Data preprocessing completed")
        print("✅ Class balance analysis completed")
        print("✅ Model training completed")
        print("✅ Model evaluation completed")
        print(f"\n🏆 Best model: Logistic Regression with {best_solver} solver")
        print(f"📊 Final accuracy: {metrics['accuracy']:.4f}")
        print(f"🎯 Final F1-score: {metrics['f1']:.4f}")
        print(f"📈 Final ROC-AUC: {metrics['roc_auc']:.4f}")
        
        print("\n📁 Generated files:")
        print("- X_train.csv, X_test.csv, y_train.csv, y_test.csv (preprocessed data)")
        print("- logistic_regression_model.pkl (trained model)")
        print("- y_pred.csv, y_pred_proba.csv (predictions)")
        
        print("\n🚀 Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
