import unittest
import json
import os
import pandas as pd
import time
from openai import APIStatusError, InternalServerError
import openai
import sys

# ✅ Ensure repo root is in sys.path so "auto_feat" is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auto_feat import AutoFeaturizer
from auto_feat.featurization_module.proposal import feat_proposal


# === Dummy LLM API Wrapper (same style as your other tests) ===
MODEL = "argo:gpt-5-mini"
client = openai.OpenAI(
    api_key="whatever+random",
    base_url="http://0.0.0.0:60963/v1",
)

def dummy_llm(prompt, model=MODEL, temperature=0.3, max_attempts=5):
    """
    Dummy LLM wrapper with retry logic.
    In real tests, mock this to return fixed code.
    """
    for attempt in range(max_attempts):
        try:
            # Always return a valid JSON string
            return json.dumps({
                "new_feature_computation": {
                    "feature_sum": "sum of A and B",
                    "feature_ratio": "ratio of A to C"
                }
            })
        except (APIStatusError, InternalServerError) as e:
            if getattr(e, "status_code", 500) >= 500:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise
        except Exception as e:
            if "unexpected mimetype" in str(e).lower():
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise
    raise RuntimeError("Dummy LLM failed after retries")


class TestProposal(unittest.TestCase):

    def setUp(self):
        # Build absolute paths for robustness
        self.state = AutoFeaturizer(
            target="dummy_target"
        )

        # Populate state attributes expected by feat_proposal
        self.state.features_description = {
            "A": "First numeric feature",
            "B": "Second numeric feature",
            "C": "Third numeric feature"
        }
        self.state.literature_review = "Paper suggests combining A and B, and using ratio A/C."
        self.state.eval_report = {
            "performance": {"test": {"RMSE": 0.15, "R2": 0.82}},
            "feature_importance": [
                {"variable": "A", "importance": 0.6},
                {"variable": "B", "importance": 0.3},
                {"variable": "C", "importance": 0.1}
            ]
        }

    def test_feat_proposal_success(self):
        agent = feat_proposal(dummy_llm, max_retries=2)
        result = agent(self.state)
        print("Feat Proposal Result:", result)

        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn("feature_sum", result)
        self.assertIn("feature_ratio", result)
        self.assertEqual(result["feature_sum"], "sum of A and B")
        self.assertEqual(result["feature_ratio"], "ratio of A to C")


if __name__ == "__main__":
    unittest.main()
