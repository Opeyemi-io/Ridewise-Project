import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn

df = pd.read_parquet('..//ridewise project dataset/model_input.parquet')

x = df.drop(columns=['churn'])
y = df['churn']

print(y.value_counts())
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

mlflow.set_experiment("Ridewise Churn Prediction")

with mlflow.start_run(run_name="Logistic Regression") as run:
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1] 
    auc_score = roc_auc_score(y_test, y_prob)
    
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("features", x.columns.tolist())
    mlflow.log_param("mi_threshold", 0.000001)
    mlflow.log_metric("roc_auc", auc_score)
    mlflow.sklearn.log_model(model, "model")

    print(f"Run ID: {run.info.run_id}")
    print(f"ROC AUC Score: {auc_score}")