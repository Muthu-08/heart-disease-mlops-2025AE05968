import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

NUM_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)
        ]
    )

def train_and_track():
    mlflow.set_experiment("heart-disease-risk-prediction")
    
    df = pd.read_csv("data/processed_heart.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = build_preprocessor()
    
    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, C=0.1, random_state=42),
            {"C": 0.1, "max_iter": 1000}
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            {"n_estimators": 100, "max_depth": 5}
        )
    }
    
    best_auc = 0.0
    best_pipeline = None

    for model_name, (clf, params) in models.items():
        with mlflow.start_run(run_name=model_name):
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf)
            ])
            
            # 5-fold cross validation
            cv_results = cross_validate(
                pipeline, X_train, y_train, cv=5, 
                scoring=["accuracy", "precision", "recall", "roc_auc"]
            )
            
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            probs = pipeline.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)
            
            # Log params & metrics
            mlflow.log_params(params)
            mlflow.log_metric("cv_mean_roc_auc", cv_results["test_roc_auc"].mean())
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_precision", prec)
            mlflow.log_metric("test_recall", rec)
            mlflow.log_metric("test_roc_auc", auc)
            
            # Use cloudpickle to avoid skops security blocks with Tree structures
            mlflow.sklearn.log_model(
                sk_model=pipeline, 
                artifact_path="model", 
                serialization_format="cloudpickle"
            )
            
            if auc > best_auc:
                best_auc = auc
                best_pipeline = pipeline

    # Ensure output directory exists before dumping with joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, "models/best_model.joblib")
    print(f"Training complete. Best model saved with Test ROC-AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train_and_track()