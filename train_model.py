import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import time

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
print("=" * 70)

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

print("\n" + "=" * 70)
print(f"Best performing solver: {best_solver} with accuracy: {best_accuracy:.4f}")

# Save the best model and predictions
joblib.dump(best_model, 'logistic_regression_model.pkl')
pd.DataFrame(best_predictions).to_csv('y_pred.csv', index=False)
pd.DataFrame(best_probabilities).to_csv('y_pred_proba.csv', index=False)

print(f"\nBest model ({best_solver}) saved as logistic_regression_model.pkl")
print("Predictions saved as y_pred.csv and y_pred_proba.csv")

# Additional information about gradient descent
print("\n" + "=" * 70)
print("GRADIENT DESCENT SOLVER DETAILS:")
print("=" * 70)
print("• lbfgs: Uses limited-memory BFGS optimization")
print("• sag: Stochastic Average Gradient - good for large datasets")
print("• saga: SAG with averaging - supports L1 regularization")
print("• newton-cg: Newton's method with conjugate gradient")
print("• liblinear: Coordinate descent (not gradient descent)")
print("\nFor pure gradient descent implementation, 'sag' or 'saga' are recommended.")


