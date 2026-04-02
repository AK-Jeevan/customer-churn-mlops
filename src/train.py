import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, recall_score

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn


# -------------------------
# MLflow setup (SAFE VERSION)
# -------------------------
try:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("churn-experiment")
    print("Using MLflow server at 127.0.0.1:5000")
except Exception:
    mlflow.set_tracking_uri("file:///./mlruns")
    mlflow.set_experiment("churn-experiment")
    print("Using local MLflow tracking")


def load_data(path):
    return pd.read_csv(path)


def main():
    with mlflow.start_run():

        df = load_data("data/customer_churn.csv")

        X = df.drop("churn", axis=1)
        y = df["churn"]

        # Column types
        cat_cols = X.select_dtypes(include=["object"]).columns
        num_cols = X.select_dtypes(exclude=["object"]).columns

        # Pipelines
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ])

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # -------------------------
        # Logistic Regression
        # -------------------------
        log_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])

        log_pipeline.fit(X_train, y_train)
        log_preds = log_pipeline.predict(X_test)

        log_recall = recall_score(y_test, log_preds)
        log_acc = accuracy_score(y_test, log_preds)

        print("\nLogistic Regression Results:")
        print(classification_report(y_test, log_preds, zero_division=0))

        # -------------------------
        # XGBoost
        # -------------------------
        xgb_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                scale_pos_weight=5,
                use_label_encoder=False,
                eval_metric="logloss"
            ))
        ])

        xgb_pipeline.fit(X_train, y_train)
        xgb_preds = xgb_pipeline.predict(X_test)

        xgb_recall = recall_score(y_test, xgb_preds)
        xgb_acc = accuracy_score(y_test, xgb_preds)

        print("\nXGBoost Results:")
        print(classification_report(y_test, xgb_preds, zero_division=0))

        # -------------------------
        # Select best model
        # -------------------------
        if xgb_recall > log_recall:
            best_model = xgb_pipeline
            best_name = "XGBoost"
            best_recall = xgb_recall
        else:
            best_model = log_pipeline
            best_name = "Logistic Regression"
            best_recall = log_recall

        print(f"\nBest Model: {best_name}")

        # -------------------------
        # MLflow logging
        # -------------------------
        mlflow.log_param("model", best_name)
        mlflow.log_metric("recall_class_1", best_recall)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model"
        )

        # -------------------------
        # Save locally
        # -------------------------
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/model.pkl")
        joblib.dump(X.columns.tolist(), "models/columns.pkl")

        print("\nModel saved successfully!")


if __name__ == "__main__":
    main()