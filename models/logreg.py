import numpy as np
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import util
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    clf.predict(r'data\ds1_valid.csv')
    
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m=x.shape[0]
        n=x.shape[1]
        self.theta=np.zeros(n)
        eps_count=0
        while(eps_count!=5):
            theta_prev=self.theta
            h = 1 / (1 + np.exp(-x @ self.theta))
            grad = 1/m*(x.T@(1/(1+np.exp(-x@self.theta))-y))
            Hessian = (1 / m) * (x.T * (h * (1 - h))) @ x
            self.theta=self.theta - np.linalg.inv(Hessian)@grad
            if ((self.theta.T@self.theta).item() - (theta_prev.T@theta_prev).item() <= self.eps):
                eps_count=eps_count+1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).
        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return np.where(1/(1+np.exp(-x@self.theta)) >= 0.5, 1, 0) 
        # *** END CODE HERE ***
    # Outside the class, at the bottom of the file
if __name__ == "__main__":
    x_train, y_train = util.load_dataset(r'data/ds1_train.csv', add_intercept=True)
    x_val, y_val = util.load_dataset(r'data/ds1_valid.csv', add_intercept=True)
    
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    
    # Use x_val (the data), not the file path string
    predictions = clf.predict(x_val)
    print(predictions)  
    accuracy = np.mean(predictions == y_val)
    print(f"Validation Accuracy: {accuracy * 100:.4f}%") 