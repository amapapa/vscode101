### Write a linear regression code using a modular approach with separate functions for model training, prediction, and evaluation.

### IMPORT all the required LIBRARIES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Model Module: Encapsulates the Linear Regression logic
class LinearRegressionCustom:
    """
    A modular Linear Regression model implemented from scratch using NumPy, 
    trained with Gradient Descent.
    """
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.W = None
        self.b = None
        self.m = None # Number of training examples
        self.n = None # Number of features

    def fit(self, X, Y):
        """Trains the linear regression model using gradient descent."""
        self.m, self.n = X.shape
        # Weight initialization
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # Gradient Descent loop
        for _ in range(self.iterations):
            self.update_weights()
        
        return self

    def update_weights(self):
        """Helper function to update weights and bias in gradient descent."""
        Y_pred = self.predict(self.X)
        # Calculate gradients
        dW = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.m

        db = - 2 * np.sum(self.Y - Y_pred) / self.m

        # Update weights and bias
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        """Predicts the target values for a given input X."""
        return X.dot(self.W) + self.b

# 2. Main Execution Module: Handles data loading, splitting, and model usage
def main():
    # Example data (replace with your own dataset loading)
    # Using a simple example similar to snippets found
    X_data = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6], dtype=float).reshape(-1, 1)
    y_data = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86], dtype=float)

    # Splitting data into training and testing sets (using a manual split for simplicity)
    X_train, X_test = X_data[:-3], X_data[-3:]
    y_train, y_test = y_data[:-3], y_data[-3:]

    # Create and fit the modular linear regression model
    model = LinearRegressionCustom(learning_rate=0.001, iterations=10000)
    model.fit(X_train, y_train)

    # Get results
    print(f"Model Intercept (b): {model.b:.2f}")
    print(f"Model Coefficient (W): {model.W[0]:.2f}")

    # Make predictions on test data
    y_pred = model.predict(X_test)
    print(f"\nActual Y values: {y_test}")
    print(f"Predicted Y values: {np.round(y_pred, 2)}")

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nMean Squared Error (MSE) on test data: {mse:.2f}")
    print(f"R-squared (R²) score on test data: {r2:.2f}")

if __name__ == "__main__":
    main()
