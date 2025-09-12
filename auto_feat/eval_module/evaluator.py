import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import uuid
import time

def create_evaluation_agent_wrap(max_retries: int = 3, task: str = "regression"):
    """
    Wraps the Evaluation Module as a node for LangGraph.

    Args:
        max_retries (int): Number of retries if training/evaluation fails.
        task (str): "regression" or "classification".
    """

    def agent_node(state: object):
        """
        state must provide:
          - state.clean_augmented_data: pandas.DataFrame (dataset with features + target)
          - state.feature_keys: list of feature names
          - state.target_key: str, target column name

        state will be updated with:
          - state.eval_report: dict (feedback report with metrics, feature importance, narrative)
        """
        df = state.clean_augmented_data
        feature_keys = state.feature_keys
        target_key = state.target_key

        for attempt in range(max_retries):
            try:
                # Extract X, y
                X = df[feature_keys]
                y = df[target_key]

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Choose model
                if task == "regression":
                    model = RandomForestRegressor(
                        n_estimators=200, random_state=42
                    )
                else:
                    model = RandomForestClassifier(
                        n_estimators=200, random_state=42
                    )

                # Fit
                model.fit(X_train, y_train)

                # Predict
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Report
                report = {
                    "model_id": f"RF_{uuid.uuid4().hex[:8]}_{int(time.time())}",
                    "model_type": f"RandomForest_{task}",
                    "performance": {"train": {}, "test": {}},
                    "feature_importance": [],
                }

                if task == "regression":
                    report["performance"]["train"] = {
                        "MSE": mean_squared_error(y_train, y_train_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
                        "R2": r2_score(y_train, y_train_pred),
                        "N Obs": len(y_train),
                    }
                    report["performance"]["test"] = {
                        "MSE": mean_squared_error(y_test, y_test_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
                        "R2": r2_score(y_test, y_test_pred),
                        "N Obs": len(y_test),
                    }
                else:
                    report["performance"]["train"] = {
                        "Accuracy": accuracy_score(y_train, y_train_pred),
                        "LogLoss": log_loss(y_train, model.predict_proba(X_train)),
                        "N Obs": len(y_train),
                    }
                    report["performance"]["test"] = {
                        "Accuracy": accuracy_score(y_test, y_test_pred),
                        "LogLoss": log_loss(y_test, model.predict_proba(X_test)),
                        "N Obs": len(y_test),
                    }

                # Feature importance
                importances = model.feature_importances_
                scaled = importances / np.max(importances)
                total = np.sum(importances)

                for var, imp, sc in zip(feature_keys, importances, scaled):
                    report["feature_importance"].append(
                        {
                            "variable": var,
                            "relative_importance": float(imp),
                            "scaled_importance": float(sc),
                            "percentage": float(imp / total) if total > 0 else 0.0,
                        }
                    )

                # Narrative
                if task == "regression":
                    top_feature = feature_keys[int(np.argmax(importances))]
                    r2 = report["performance"]["test"]["R2"]
                    report["narrative"] = (
                        f"RandomForestRegressor achieved R²={r2:.3f} on test data. "
                        f"Top feature: {top_feature}."
                    )
                else:
                    top_feature = feature_keys[int(np.argmax(importances))]
                    acc = report["performance"]["test"]["Accuracy"]
                    report["narrative"] = (
                        f"RandomForestClassifier achieved Accuracy={acc:.3f} on test data. "
                        f"Top feature: {top_feature}."
                    )

                # Update state
                state.eval_report = report
                return

            except Exception as e:
                print(f"❌ Evaluation attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Evaluation failed after {max_retries} attempts. Last error: {e}"
                    )
                continue

    return agent_node
