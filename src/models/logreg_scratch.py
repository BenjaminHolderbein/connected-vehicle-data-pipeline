import numpy as np
"""
From-scratch Logistic Regression model.

Implements binary logistic regression using full-batch gradient descent
with binary cross-entropy loss. Designed with a minimal API to resemble
scikit-learn:

    model = Model(lr=0.1, epochs=1000)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)

Attributes:
    name (str): Identifier string for registry usage ("logreg_scratch").
    lr (float): Learning rate for gradient descent.
    epochs (int): Maximum number of passes through the data.
    tol (float): Convergence tolerance for early stopping.
    random_state (int): RNG seed for reproducibility.
    verbose (bool): If True, print loss during training.
    w (np.ndarray): Learned weight vector, including bias as last term.
"""


class Model:
    """
    Logistic Regression (scratch implementation).

    Methods:
        fit(X, y):
            Train the model with full-batch gradient descent.
        predict_proba(X):
            Return predicted probabilities for the positive class.
    """
    name = "logreg_scratch"

    def __init__(self, lr: float = 0.1, epochs: int = 1000, tol: float = 1e-6,
                 random_state: int = 123, verbose: bool = False):
        """
        Initialize model.

        Args:
            lr (float): Learning rate.
            epochs (int): Maximum training iterations.
            tol (float): Convergence tolerance for early stopping.
            random_state (int): Seed for reproducibility.
            verbose (bool): Whether to print progress during training.
        """
        self.lr = lr
        self.epochs = epochs
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.w = None  # includes bias as last term

    @staticmethod
    def _sigmoid(z):
        """
        Compute the sigmoid function element-wise.

        Args:
            z (np.ndarray): Input array.

        Returns:
            np.ndarray: Values in (0, 1).
        """
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model":
        """
        Fit the logistic regression model using gradient descent.

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features).
            y (np.ndarray): Binary labels (0 or 1), shape (n_samples,).

        Returns:
            Model: Fitted model instance (self).
        """
        X = np.asarray(X)
        y = np.asarray(y).astype(float)
        Xb = np.c_[X, np.ones((X.shape[0], 1))]  # (n, d+1)
        n, d1 = Xb.shape

        rng = np.random.default_rng(self.random_state)
        # Small random init; last weight is bias term
        self.w = rng.normal(scale=0.01, size=d1)

        prev_loss = np.inf
        for t in range(self.epochs):
            z = Xb @ self.w              # (n,)
            p = self._sigmoid(z)         # (n,)
            # Binary cross-entropy (mean)
            eps = 1e-12
            bce = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()

            # Gradient (vectorized)
            error = (p - y)              # (n,)
            grad = (Xb.T @ error) / n    # (d+1,)

            # Update (gradient descent)
            self.w -= self.lr * grad

            if self.verbose and (t % 100 == 0 or t == self.epochs - 1):
                print(f"[{t}] loss={bce:.6f}")

            if abs(prev_loss - bce) < self.tol:
                break
            prev_loss = bce

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for the positive class.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probabilities of class 1, shape (n_samples,).
        """
        X = np.asarray(X)
        Xb = np.c_[X, np.ones((X.shape[0], 1))]
        return self._sigmoid(Xb @ self.w)  # (n,)
