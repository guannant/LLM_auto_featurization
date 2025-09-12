"""
Utility definitions for the featurization module
"""


class PreviousRunsReports():
    """
    Stores reports of previous runs and features used
    """
    def __init__(self):
        pass


import h2o
from h2o.estimators import H2OGradientBoostingEstimator

h2o.init()


df = h2o.import_file("https://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris_wheader.csv")

train, test = df.split_frame(ratios=[0.8])
x = df.columns[:-1]
y = "class"

model = H2OGradientBoostingEstimator()
model.train(x=x, y=y, training_frame=train)


train_perf = model.model_performance(train)

test_perf = model.model_performance(test)


feat_importance = model.varimp(use_pandas=True)
print(feat_importance)



###########################################################################

def create_short_h2o_report(model, train_perf, test_perf):
    report = {
        "model_id": model.model_id,
        "model_type": type(model).__name__,
        "train_metrics": train_perf._metric_json,
        "test_metrics": test_perf._metric_json,
        "feature_importance": model.varimp(use_pandas=True).to_dict(orient="records")
    }
    return report

short_report = create_short_h2o_report(model, train_perf, test_perf)


##################################################################################


def summarize_h2o_report(report: dict) -> dict:
    """
    Create a simplified, LLM-friendly report from raw H2O model output.
    """

    def simplify_metrics(metrics):
        return {
            "MSE": metrics.get("MSE"),
            "RMSE": metrics.get("RMSE"),
            "R2": metrics.get("r2"),
            "LogLoss": metrics.get("logloss"),
            "Accuracy": 1 - metrics.get("mean_per_class_error", None)
            if metrics.get("mean_per_class_error") is not None
            else None,
            "Mean Per-Class Error": metrics.get("mean_per_class_error"),
            "N Obs": metrics.get("nobs")
        }

    clean_report = {
        "model_id": report["model_id"],
        "model_type": report["model_type"],
        "performance": {
            "train": simplify_metrics(report["train_metrics"]),
            "test": simplify_metrics(report["test_metrics"]),
        },
        "feature_importance": sorted(
            report["feature_importance"], key=lambda x: x["relative_importance"], reverse=True
        ),
    }

    top_feat = clean_report["feature_importance"][0]["variable"]
    clean_report["narrative"] = (
        f"Model {clean_report['model_type']} achieved R²={clean_report['performance']['test']['R2']:.3f} "
        f"and accuracy≈{clean_report['performance']['test']['Accuracy']:.2%} on test data. "
        f"Top feature: {top_feat}."
    )

    return clean_report

summarize_h2o_report(short_report)


