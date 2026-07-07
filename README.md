About MLFLOW

1) Mlflow is used to document and track various versions of trained model that has been pushed to production, giving opportunity to compare versions, evaluate  how drift patterns impacts model performance

2) For the purpose of tracking in MLflow, permits defining tracking hyperparameters, metrics and models

3) For the purpose of evaluation, has a robust evaluation framework that goes beyond simple accuracy metrics, providing deep insights into model behavior, performance characteristics, and real-world readiness through automated testing, visualization, and validation pipelines.(structures for model, function, Dataset,custom metrics and visualization, shap intergration and plugin evaluators)

4) For the purpose of deploying, Mlflow allows serving models to various targets both local and cloud i.e from dev to production environment

5) The Model Registry creates a pipeline between tracking and deployment as a typical model doesn't go straight from a training script to production; it gets registered, assigned a version number, and moved through stages (None → Staging → Production → Archived), giving a clear audit trail of what changed and when.

6) MLflow is framework-agnostic because it works the same way whether the model is scikit-learn, XGBoost, PyTorch, TensorFlow, or even a custom Python function, via a unified "flavor" system, so teams aren't locked into one ML library just to get tracking benefits.

8) Every run automatically logs "artifacts" beyond just metrics — this includes the model file itself, plots, config files, and even the exact conda/pip environment used, which is what actually makes a run reproducible months later, not just the metric numbers.

9) MLflow supports autologging — for popular libraries (scikit-learn, XGBoost, LightGBM), calling mlflow.autolog() automatically records hyperparameters, metrics, and the model with zero manual logging code, which lowers the barrier to consistent tracking across a team.

