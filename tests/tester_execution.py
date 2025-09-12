import unittest
import pandas as pd
import os
import time
import openai
from openai import APIStatusError, InternalServerError
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === Import from project ===
from auto_feat import AutoFeaturizer
from auto_feat.featurization_module.execution import feature_generation
from auto_feat.first_pass.summarization.summarize import summarize


# === Dummy LLM wrapper ===
MODEL = "argo:gpt-5-mini"
client = openai.OpenAI(
    api_key="whatever+random",
    base_url="http://0.0.0.0:60963/v1",
)

def dummy_llm(prompt, model=MODEL, temperature=0.3, max_attempts=5):
    """
    A thin wrapper around OpenAI client with retry logic.
    In real tests, you should mock this to return fixed code.
    """
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
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
    raise RuntimeError("ChatCompletion failed after retries")



class TestFeatureGeneration(unittest.TestCase):

    def test_basic_generation(self):
        state = AutoFeaturizer(
            target="dummy_target"
        )
        state.construct_strategy = {
            "feature_sum": "sum of A and B",
            "feature_ratio": "ratio of A and B"
        }
        state.feature_dict = state.construct_strategy
        state.clean_augmented_data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

        agent = feature_generation(dummy_llm, max_retries=2)
        agent(state)

        self.assertIn("feature_sum", state.clean_augmented_data.columns)

    def test_missing_values(self):
        state = AutoFeaturizer("dummy_target")
        state.construct_strategy = {"feature_sum": "sum of A and B"}
        state.feature_dict = state.construct_strategy
        state.clean_augmented_data = pd.DataFrame({"A": [1, None, 3], "B": [4, 5, None]})

        agent = feature_generation(dummy_llm, max_retries=2)
        agent(state)

        self.assertTrue(state.clean_augmented_data["feature_sum"].isna().any())

    def test_division_by_zero(self):
        state = AutoFeaturizer("dummy_target")
        state.construct_strategy = {"feature_ratio": "ratio of A to C"}
        state.feature_dict = state.construct_strategy
        state.clean_augmented_data = pd.DataFrame({"A": [1, 2, 3], "C": [0, 1, 0]})

        agent = feature_generation(dummy_llm, max_retries=2)
        agent(state)

        self.assertTrue(
            (state.clean_augmented_data["feature_ratio"].isin([float("inf"), float("-inf")]) |
             state.clean_augmented_data["feature_ratio"].isna()).any()
        )

    def test_invalid_column_reference(self):
        state = AutoFeaturizer("dummy_target")
        state.construct_strategy = {"feature_invalid": "sum of A and Z (Z does not exist)"}
        state.feature_dict = state.construct_strategy
        state.clean_augmented_data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

        agent = feature_generation(dummy_llm, max_retries=2)
        agent(state)

        self.assertIn("feature_invalid", state.clean_augmented_data.columns)
        self.assertTrue(state.clean_augmented_data["feature_invalid"].isna().all())


if __name__ == "__main__":
    unittest.main()
