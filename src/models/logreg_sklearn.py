"""
Logistic Regression (scikit-learn wrapper).

Provides a thin wrapper around sklearn's LogisticRegression
with a simplified API consistent with other models in this project.
"""


from sklearn.linear_model import LogisticRegression


class Model:
    """
    Logistic Regression model using scikit-learn.

    Attributes:
        name (str): Identifier for registry usage ("logreg").
        clf (sklearn.linear_model.LogisticRegression):
            The underlying scikit-learn classifier instance.
    """
    name = "logreg"

    def __init__(self, random_state: int = 123):
        """
        Initialize model.

        Args:
            random_state (int, optional): Seed for reproducibility.
        """
        self.clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        )

    def fit(self, X, y):
        """
        Fit the logistic regression model.

        Args:
            X (array-like): Training features of shape (n_samples, n_features).
            y (array-like): Training labels of shape (n_samples,).

        Returns:
            Model: Fitted model instance (self).
        """
        return self.clf.fit(X, y)

    def predict_proba(self, X):
        """
        Predict probabilities for the positive class.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probabilities for class 1, shape (n_samples,).
        """
        # Return 1-class probability as a 1D array
        return self.clf.predict_proba(X)[:, 1]
