from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import CombinedFeatures

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "SMSSpamCollection"
MODEL_PATH = PROJECT_ROOT / "models" / "final_spam_pipeline.pkl"

def main():
    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
    )
    df = df.drop_duplicates().copy()

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    pipeline = Pipeline(
    [
        ("features", CombinedFeatures()),
        (
            "model",
            LogisticRegression(
                C=50,
                class_weight="balanced",
            ),
        ),
    ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    print(classification_report(y_test, y_pred))
    joblib.dump(pipeline, MODEL_PATH)
    print()
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
