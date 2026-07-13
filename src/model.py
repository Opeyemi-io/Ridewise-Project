from pathlib import Path

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    import mlflow.xgboost

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "ridewise project dataset" / "model_input.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENT_NAME = "Ridewise Churn Prediction"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Load preprocessed training data.
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Missing model input file: {DATA_PATH}. Run src/preprocessing.py first."
    )

df = pd.read_parquet(DATA_PATH)

# Step 2: Build features and target, then split train/test.
x = df.drop(columns=["churn"])
y = df["churn"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print(y_train.value_counts())
print(y_train.shape)

# Step 3: Group all runs under one experiment name.
mlflow.set_experiment(EXPERIMENT_NAME)

# Step 4: Logistic Regression run.
with mlflow.start_run(run_name="Logistic Regression") as run:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, class_weight="balanced", max_iter=3000)),
        ]
    )
    model.fit(x_train, y_train)

    y_prob = model.predict_proba(x_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("n_features", x.shape[1])
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_metric("roc_auc", auc_score)
    mlflow.sklearn.log_model(model, name="model")

    joblib.dump(model, MODELS_DIR / "logistic_regression.joblib")

    print(f"Run ID: {run.info.run_id}")
    print(f"Logistic Regression ROC AUC: {auc_score:.4f}")

# Step 5: Random Forest with hyperparameter search.
param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 3],
}

with mlflow.start_run(run_name="Random Forest") as run:
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_distributions=param_dist,
        scoring="roc_auc",
        cv=5,
        n_iter=10,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_

    y_prob = model.predict_proba(x_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)

    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_features", x.shape[1])
    mlflow.log_param("search_cv", 5)
    mlflow.log_metric("best_cv_roc_auc", search.best_score_)
    mlflow.log_params({f"best_{key}": value for key, value in search.best_params_.items()})
    mlflow.log_metric("roc_auc", auc_score)
    mlflow.sklearn.log_model(model, name="model")

    feature_importance = (
        pd.Series(model.feature_importances_, index=x.columns)
        .sort_values(ascending=False)
        .rename("importance")
        .rename_axis("feature")
        .reset_index()
    )
    rf_importance_path = MODELS_DIR / "feature_importance_random_forest.csv"
    feature_importance.to_csv(rf_importance_path, index=False)
    mlflow.log_artifact(str(rf_importance_path))
    joblib.dump(model, MODELS_DIR / "random_forest.joblib")

    print(f"Run ID: {run.info.run_id}")
    print(f"Random Forest ROC AUC: {auc_score:.4f}")
    print(f"Best params: {search.best_params_}")

# Step 6: Optional XGBoost run (only if xgboost is installed).
if HAS_XGBOOST:
    with mlflow.start_run(run_name="XGBoost") as run:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(x_train, y_train)

        y_prob = model.predict_proba(x_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_prob)

        mlflow.log_param("model", "XGBClassifier")
        mlflow.log_param("n_features", x.shape[1])
        mlflow.log_metric("roc_auc", auc_score)
        mlflow.xgboost.log_model(model, name="model")

        feature_importance = (
            pd.Series(model.feature_importances_, index=x.columns)
            .sort_values(ascending=False)
            .rename("importance")
            .rename_axis("feature")
            .reset_index()
        )
        xgb_importance_path = MODELS_DIR / "feature_importance_xgboost.csv"
        feature_importance.to_csv(xgb_importance_path, index=False)
        mlflow.log_artifact(str(xgb_importance_path))
        joblib.dump(model, MODELS_DIR / "xgboost.joblib")

        print(f"Run ID: {run.info.run_id}")
        print(f"XGBoost ROC AUC: {auc_score:.4f}")
else:
    print("XGBoost not installed. Skipping XGBoost run.")
