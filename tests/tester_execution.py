import unittest
import pandas as pd
from openai import APIStatusError, InternalServerError
import openai
import time

from execution import feature_generation  # replace with actual filename

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


class State:
    """Simple container for passing state into the agent."""
    pass


class TestSummarizer(unittest.TestCase):

    def test_summarizer(self):
        state = State()

        state.data_path = "data.csv"  # replace with actual path
        state.manuscript_path = "manuscript.txt"  # replace with actual path

        agent = summarize(dummy_llm, max_retries=5)
        success = agent(state)
        # self.assertIn("feature_sum", state.clean_augmented_data.columns)


class TestFeatureGeneration(unittest.TestCase):
    
    def test_basic_generation(self):
        state = State()
        state.construct_strategy = {
            "feature_sum": "sum of A and B",
            "feature_ratio": "ratio of A and C"
        }
        state.feature_dict = state.construct_strategy
        state.clean_augmented_data = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})

        agent = feature_generation(dummy_llm, max_retries=5)
        success = agent(state)

    
        self.assertIn("feature_sum", state.clean_augmented_data.columns)
        self.assertIn("feature_ratio", state.clean_augmented_data.columns)

    def test_missing_values(self):
        state = State()
        state.construct_strategy = {"feature_sum": "sum of A and B"}
        state.feature_dict = state.construct_strategy
        state.clean_augmented_data = pd.DataFrame({"A": [1, None, 3], "B": [4, 5, None]})

        agent = feature_generation(dummy_llm, max_retries=5)
        success = agent(state)


        self.assertTrue(state.clean_augmented_data["feature_sum"].isna().any())

    def test_division_by_zero(self):
        state = State()
        state.construct_strategy = {"feature_ratio": "ratio of A to C"}
        state.feature_dict = state.construct_strategy
        state.clean_augmented_data = pd.DataFrame({"A": [1, 2, 3], "C": [0, 1, 0]})

        agent = feature_generation(dummy_llm, max_retries=5)
        success = agent(state)


        self.assertTrue(
            (state.clean_augmented_data["feature_ratio"].isin([float("inf"), float("-inf")]) |
             state.clean_augmented_data["feature_ratio"].isna()).any()
        )

    def test_invalid_column_reference(self):
        state = State()
        state.construct_strategy = {"feature_invalid": "sum of A and Z (Z does not exist)"}
        state.feature_dict = state.construct_strategy
        state.clean_augmented_data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

        agent = feature_generation(dummy_llm, max_retries=3)
        success = agent(state)

        # ✅ Now it should succeed, with the invalid feature filled as NaN

        self.assertIn("feature_invalid", state.clean_augmented_data.columns)
        self.assertTrue(state.clean_augmented_data["feature_invalid"].isna().all())


if __name__ == "__main__":
    unittest.main()
