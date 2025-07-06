import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

class LogisticRegressionFromScratch:
    """
    Logistic Regression implementation from scratch using gradient descent.
    
    The logistic regression model predicts the probability of a binary outcome
    using the logistic (sigmoid) function: p = 1 / (1 + e^(-z))
    where z = w0 + w1*x1 + w2*x2 + ... + wn*xn
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Initialize the logistic regression model.
        
        Parameters:
        - learning_rate: Step size for gradient descent (controls how fast we learn)
        - max_iterations: Maximum number of training iterations
        - tolerance: Convergence threshold (stop when cost change is smaller than this)
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None  # Model parameters (w0, w1, w2, ..., wn)
        self.bias = None     # Intercept term (w0)
        self.cost_history = []  # Track cost function values during training
        
    def _add_intercept(self, X):
        """
        Add bias column (column of ones) to the feature matrix.
        This allows us to include the intercept term in our linear combination.
        
        X shape: (m, n) -> returns shape: (m, n+1)
        where m = number of samples, n = number of features
        """
        # np.ones creates a column vector of ones with same number of rows as X
        intercept = np.ones((X.shape[0], 1))
        # np.concatenate combines the intercept column with original features
        return np.concatenate((intercept, X), axis=1)
    
    def _sigmoid(self, z):
        """
        Sigmoid activation function: σ(z) = 1 / (1 + e^(-z))
        
        This function maps any real number to a value between 0 and 1,
        making it perfect for binary classification probabilities.
        
        Properties:
        - σ(0) = 0.5 (decision boundary)
        - σ(+∞) → 1 (high confidence for positive class)
        - σ(-∞) → 0 (high confidence for negative class)
        """
        # Clip z to prevent overflow in exponential function
        # Large positive z values would cause exp(-z) to underflow to 0
        # Large negative z values would cause exp(-z) to overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _cost_function(self, h, y):
        """
        Calculate the logistic regression cost function (log-likelihood).
        
        Cost = -1/m * Σ[y*log(h) + (1-y)*log(1-h)]
        
        This is the negative log-likelihood of the data given the model.
        - When y=1: cost = -log(h) -> penalizes low probability predictions
        - When y=0: cost = -log(1-h) -> penalizes high probability predictions
        
        Parameters:
        - h: predicted probabilities from sigmoid function
        - y: true binary labels (0 or 1)
        """
        # Add small epsilon to prevent log(0) which would cause -inf
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)
        
        # Calculate cost using vectorized operations
        # y * log(h): cost when true label is 1
        # (1-y) * log(1-h): cost when true label is 0
        cost = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost
    
    def _gradient_descent(self, X, h, y):
        """
        Calculate gradients for gradient descent optimization.
        
        The gradient of the cost function with respect to weights is:
        ∂J/∂w = 1/m * X^T * (h - y)
        
        This tells us the direction and magnitude to update weights
        to minimize the cost function.
        
        Parameters:
        - X: feature matrix with intercept column
        - h: predicted probabilities
        - y: true labels
        """
        # Number of training examples
        m = X.shape[0]
        
        # Calculate gradient: X^T * (predictions - actual)
        # This gives us the direction to move each weight
        gradient = np.dot(X.T, (h - y)) / m
        return gradient
    
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.
        
        Algorithm:
        1. Initialize weights randomly
        2. For each iteration:
           a. Calculate predictions using current weights
           b. Calculate cost (how wrong we are)
           c. Calculate gradients (direction to improve)
           d. Update weights in direction that reduces cost
        3. Stop when cost stops improving significantly
        """
        # Convert to numpy arrays for mathematical operations
        X = np.array(X)
        y = np.array(y)
        
        # Add intercept term (bias) to features
        X = self._add_intercept(X)
        
        # Initialize weights randomly with small values
        # Shape: (n_features + 1,) to include intercept
        np.random.seed(42)  # For reproducible results
        self.weights = np.random.normal(0, 0.01, X.shape[1])
        
        # Training loop - gradient descent optimization
        for i in range(self.max_iterations):
            # Forward pass: calculate predictions
            # z = X * weights (linear combination of features)
            z = np.dot(X, self.weights)
            
            # Apply sigmoid to get probabilities between 0 and 1
            h = self._sigmoid(z)
            
            # Calculate cost (how wrong our predictions are)
            cost = self._cost_function(h, y)
            self.cost_history.append(cost)
            
            # Calculate gradients (direction to improve weights)
            gradient = self._gradient_descent(X, h, y)
            
            # Update weights: move in direction that reduces cost
            # weights = weights - learning_rate * gradient
            self.weights -= self.learning_rate * gradient
            
            # Check for convergence (stop if improvement is very small)
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
        
        print(f"Training completed after {len(self.cost_history)} iterations")
        print(f"Final cost: {self.cost_history[-1]:.6f}")
    
    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.
        
        Returns probabilities between 0 and 1 for each sample.
        """
        # Convert to numpy array and add intercept
        X = np.array(X)
        X = self._add_intercept(X)
        
        # Calculate linear combination: z = X * weights
        z = np.dot(X, self.weights)
        
        # Apply sigmoid to get probabilities
        probabilities = self._sigmoid(z)
        return probabilities
    
    def predict(self, X, threshold=0.5):
        """
        Make binary predictions based on probability threshold.
        
        If probability >= threshold: predict class 1
        If probability < threshold: predict class 0
        
        Default threshold is 0.5 (decision boundary)
        """
        # Get probabilities first
        probabilities = self.predict_proba(X)
        
        # Convert probabilities to binary predictions
        predictions = (probabilities >= threshold).astype(int)
        return predictions
    
    def plot_cost_history(self):
        """
        Plot the cost function over training iterations.
        This helps visualize if the model is learning properly.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history, 'b-', linewidth=2)
        plt.title('Cost Function During Training', fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost (Log-Likelihood)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance based on weight magnitudes.
        Larger absolute weights indicate more important features.
        """
        if self.weights is None:
            print("Model not trained yet!")
            return None
        
        # Skip the first weight (intercept/bias term)
        feature_weights = self.weights[1:]
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(feature_weights))]
        
        # Create DataFrame for easy visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Weight': feature_weights,
            'Abs_Weight': np.abs(feature_weights)
        }).sort_values('Abs_Weight', ascending=False)
        
        return importance_df

def demonstrate_logistic_regression():
    """
    Demonstrate the logistic regression implementation with the heart attack dataset.
    """
    print("=" * 60)
    print("LOGISTIC REGRESSION FROM SCRATCH DEMONSTRATION")
    print("=" * 60)
    
    # Load the preprocessed data
    print("\n1. Loading preprocessed data...")
    try:
        X_train = pd.read_csv("X_train.csv")
        X_test = pd.read_csv("X_test.csv")
        y_train = pd.read_csv("y_train.csv").values.ravel()
        y_test = pd.read_csv("y_test.csv").values.ravel()
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
    except FileNotFoundError:
        print("Preprocessed data files not found. Please run preprocess_data.py first.")
        return
    
    # Feature scaling is important for gradient descent
    print("\n2. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled using StandardScaler (mean=0, std=1)")
    
    # Create and train the model
    print("\n3. Training logistic regression model...")
    model = LogisticRegressionFromScratch(
        learning_rate=0.01,    # Learning rate - controls step size
        max_iterations=1000,   # Maximum training iterations
        tolerance=1e-6         # Convergence threshold
    )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    print("\n4. Making predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Get prediction probabilities
    y_train_proba = model.predict_proba(X_train_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Detailed evaluation
    print("\n5. Detailed Model Evaluation:")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))
    
    # Confusion Matrix
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Feature importance
    print("\n6. Feature Importance (Top 10):")
    feature_importance = model.get_feature_importance(X_train.columns)
    print(feature_importance.head(10))
    
    # Plot cost history
    print("\n7. Plotting cost function history...")
    model.plot_cost_history()
    
    # Save predictions
    print("\n8. Saving predictions...")
    pd.DataFrame(y_test_pred, columns=['Predicted']).to_csv("y_pred.csv", index=False)
    pd.DataFrame(y_test_proba, columns=['Probability']).to_csv("y_pred_proba.csv", index=False)
    print("Predictions saved to y_pred.csv and y_pred_proba.csv")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    # Run the demonstration
    demonstrate_logistic_regression() 