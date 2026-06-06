import mlflow
import mlflow.sklearn
import argparse
from train import main

mlflow.set_experiment("rabies-risk-pipeline")

seeds = [13, 21, 42, 87, 100]

for seed in seeds:
    with mlflow.start_run(run_name=f"logreg-seed-{seed}"):
        clf, scaler, auc, acc = main(seed)
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("seed", seed)
        mlflow.log_param("test_size", 0.25)
        mlflow.log_param("max_iter", 200)
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")

print("\nAll runs logged. Launch UI with: mlflow ui")