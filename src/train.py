"""
Train and evaluate models on the Connected Vehicle dataset.

This script loads synthetic transaction data from Postgres,
applies feature engineering, trains a chosen model, evaluates it,
and (optionally) saves the trained pipeline.

Usage:
    python src/train.py --model logreg --test-size 0.25 --threshold 0.5 --save models/logreg.pkl

Args:
    --model (str): Model identifier. Options: 'logreg', 'logreg_scratch' (default: logreg).
    --test-size (float): Fraction of data for validation split (default: 0.25).
    --threshold (float): Probability cutoff for binary classification (default: 0.5).
    --seed (int): Random seed for reproducibility (default: 123).
    --save (str): Path to save the trained pipeline as a .pkl file.

Outputs:
    - Console logs: fraud rates, ROC AUC, PR AUC, classification report.
    - Optional saved pipeline (.pkl) for later inference.
"""
# -- Imports --
import argparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

from src.data.fetch import load_training_frame
from src.features.build import add_basic_features, NUM_COLS, CAT_COLS, DROP_COLS
from src.models.preprocess import make_preprocessor
from src.eval.metrics import evaluate


def import_model(name):
    match name:
        case "logreg":
            from src.models.logreg_sklearn import Model
        case "logreg_scratch":
            from src.models.logreg_scratch import Model
        # placeholders for future models:
        # case "rf":
        #     from src.models.rf_sklearn import Model
        # case "xgb":
        #     from src.mdoels.xgb_classifier import Model
        case _:
            raise ValueError(f"Unknown model '{name}'")
    return Model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logreg")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--save", type=str, default=None)
    args = ap.parse_args()

    # Load data
    df = load_training_frame()
    y = df["is_fraud"].astype(int).values

    # Features
    df = add_basic_features(df)
    X = df.drop(columns=DROP_COLS)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"Fraud rate train/test: {y_train.mean():.3%} / {y_val.mean():.3%}")

    # Pipeline
    pre = make_preprocessor(NUM_COLS, CAT_COLS)
    Model = import_model(args.model)
    model = Model(random_state=args.seed)
    pipe = Pipeline([("pre", pre), ("clf", model)])

    # Train
    pipe.fit(X_train, y_train)
    print(f"Fitted model: {args.model}")

    # Evaluate
    proba = pipe.predict_proba(X_val)
    m = evaluate(y_val, proba, threshold=args.threshold)
    print(f"ROC AUC: {m['roc_auc']:.3f}")
    print(f"PR  AUC: {m['pr_auc']:.3f}")
    print("\nClassification report @ threshold")
    print(m["report"])

    # Save
    if args.save:
        joblib.dump(pipe, args.save)
        print(f"Saved pipeline -> {args.save}")


if __name__ == "__main__":
    main()
