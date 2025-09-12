import h2o
from h2o.estimators import H2OGradientBoostingEstimator
import numpy as np
import uuid
import time
 
 
h2o.init()
h2o.no_progress()

# Suppress Python warnings
import warnings
warnings.filterwarnings("ignore")
 
def create_evaluation_agent_wrap(max_retries: int = 3, task: str = "regression"):
    """
    Wraps the Evaluation Module using H2O models.
 
    Args:
        max_retries (int): Number of retries if training/evaluation fails.
        task (str): "regression" or "classification".
    """
 
    def agent_node(state: object):
        """
        state must provide:
          - state.clean_augmented_data: pandas.DataFrame (dataset with features + target)
          - state.cur_feature_keys: list of feature names
          - state.target: str, target column name
 
        state will be updated with:
          - state.eval_report: dict (feedback report with metrics, feature importance, narrative)
        """
        df = state.clean_augmented_data
        feature_keys = state.cur_feature_keys
        target_key = state.target
 
        for attempt in range(max_retries):
            try:
                # Convert pandas DataFrame → H2O Frame
                hf = h2o.H2OFrame(df)
 
                # Train/test split
                train, test = hf.split_frame(ratios=[0.8], seed=42)
 
                # Choose model
                if task == "regression":
                    model = H2OGradientBoostingEstimator(
                        ntrees=200,
                        seed=42
                    )
                else:
                    model = H2OGradientBoostingEstimator(
                        ntrees=200,
                        seed=42,
                        distribution="multinomial"
                    )
 
                # Train
                model.train(x=feature_keys, y=target_key, training_frame=train)
 
                # Predictions
                train_pred = model.predict(train).as_data_frame().values.flatten()
                test_pred = model.predict(test).as_data_frame().values.flatten()
 
                # Actual values
                y_train = train[target_key].as_data_frame().values.flatten()
                y_test = test[target_key].as_data_frame().values.flatten()
 
                # Report dictionary
                report = {
                    "model_id": f"H2O_GBM_{uuid.uuid4().hex[:8]}_{int(time.time())}",
                    "model_type": f"H2O_GBM_{task}",
                    "performance": {"train": {}, "test": {}},
                    "feature_importance": [],
                }
 
                # Collect performance
                perf_train = model.model_performance(train)
                perf_test = model.model_performance(test)
 
                if task == "regression":
                    report["performance"]["train"] = {
                        "MSE": perf_train.mse(),
                        "RMSE": perf_train.rmse(),
                        "R2": perf_train.r2(),
                        "N Obs": train.nrows,
                    }
                    report["performance"]["test"] = {
                        "MSE": perf_test.mse(),
                        "RMSE": perf_test.rmse(),
                        "R2": perf_test.r2(),
                        "N Obs": test.nrows,
                    }
                else:
                    report["performance"]["train"] = {
                        "Accuracy": perf_train.accuracy()[0][1],
                        "LogLoss": perf_train.logloss(),
                        "N Obs": train.nrows,
                    }
                    report["performance"]["test"] = {
                        "Accuracy": perf_test.accuracy()[0][1],
                        "LogLoss": perf_test.logloss(),
                        "N Obs": test.nrows,
                    }
 
                # Feature importance
                varimp = model.varimp(use_pandas=True)
                total = varimp["relative_importance"].sum()
 
                for _, row in varimp.iterrows():
                    report["feature_importance"].append(
                        {
                            "variable": row["variable"],
                            "relative_importance": float(row["relative_importance"]),
                            "scaled_importance": float(row["scaled_importance"]),
                            "percentage": float(row["relative_importance"] / total) if total > 0 else 0.0,
                        }
                    )
 
                if task == "regression":
                    r2 = report["performance"]["test"]["R2"]
                    top_feature = varimp.iloc[0]["variable"]
                    report["narrative"] = (
                        f"H2O GradientBoosting (regression) achieved R²={r2:.3f} on test data. "
                        f"Top feature: {top_feature}."
                    )
                else:
                    acc = report["performance"]["test"]["Accuracy"]
                    top_feature = varimp.iloc[0]["variable"]
                    report["narrative"] = (
                        f"H2O GradientBoosting (classification) achieved Accuracy={acc:.3f} on test data. "
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