import unittest
import pandas as pd
import numpy as np
import os
import sys

# ✅ Ensure repo root is in sys.path so "auto_feat" is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auto_feat.eval_module.evaluator import create_evaluation_agent_wrap
from auto_feat import AutoFeaturizer


class TestEvaluator(unittest.TestCase):

    def setUp(self):
        # Create dummy dataset
        df = pd.DataFrame({
            "feat1": np.random.rand(100),
            "feat2": np.random.rand(100),
            "target": np.random.rand(100) * 10,  # regression target
        })

        # Save to CSV in a test data folder
        base_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(base_dir, exist_ok=True)
        data_path = os.path.join(base_dir, "data.csv")
        df.to_csv(data_path, index=False)

        # Init state (AutoFeaturizer)
        self.state = AutoFeaturizer(
            target="target",
            manuscript_path=os.path.join(base_dir, "manuscript.txt"),
            data_path=data_path,
        )

        # Add feature/target keys for evaluator
        self.state.feature_keys = ["feat1", "feat2"]
        self.state.target_key = "target"

    def test_regression_evaluation(self):
        agent = create_evaluation_agent_wrap(task="regression")
        agent(self.state)

        report = self.state.eval_report
        self.assertIsInstance(report, dict)
        self.assertIn("performance", report)
        self.assertIn("feature_importance", report)
        self.assertIn("narrative", report)

        print("\n=== Evaluation Report (Regression) ===")
        print(report["narrative"])
        print(report)

    def test_classification_evaluation(self):
        # Convert target into binary classification
        self.state.clean_augmented_data["target_class"] = (
            self.state.clean_augmented_data["target"] > 5
        ).astype(int)
        self.state.target_key = "target_class"

        agent = create_evaluation_agent_wrap(task="classification")
        agent(self.state)

        report = self.state.eval_report
        self.assertIsInstance(report, dict)
        self.assertIn("performance", report)
        self.assertIn("feature_importance", report)
        self.assertIn("narrative", report)

        print("\n=== Evaluation Report (Classification) ===")
        print(report["narrative"])
        print(report)


if __name__ == "__main__":
    unittest.main()
