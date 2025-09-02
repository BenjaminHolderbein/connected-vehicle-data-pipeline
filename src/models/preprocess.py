"""
Preprocessing utilities for machine learning models.

Defines a function to create a scikit-learn ColumnTransformer
that standardizes numeric features and one-hot encodes categorical features.
"""


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def make_preprocessor(num_cols, cat_cols):
    """
    Build a preprocessing pipeline for numeric and categorical columns.

    Args:
        num_cols (list of str): Names of numeric feature columns.
        cat_cols (list of str): Names of categorical feature columns.

    Returns:
        sklearn.compose.ColumnTransformer: Transformer that applies:
            - StandardScaler() to numeric columns
            - OneHotEncoder(handle_unknown="ignore", sparse_output=False) to categorical columns
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )
