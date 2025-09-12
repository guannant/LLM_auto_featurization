import unittest
import os
import pandas as pd
import time
from openai import APIStatusError, InternalServerError
import openai

from auto_feat.first_pass.summarization.summarize import summarize
from auto_feat import AutoFeaturizer


# ---- Your LLM client ----
MODEL = "argo:gpt-5-mini"
client = openai.OpenAI(
    api_key="whatever+random",  # replace with your actual key if needed
    base_url="http://0.0.0.0:60963/v1",
)

def dummy_llm(prompt, model=MODEL, temperature=0.3, max_attempts=5):
    """
    Your thin wrapper around OpenAI client with retry logic.
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


class TestSummarizer(unittest.TestCase):

    def setUp(self):
        # Ensure data folder exists
        base_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(base_dir, exist_ok=True)

        self.manuscript_path = os.path.join(base_dir, "manuscript.txt")
        self.data_path = os.path.join(base_dir, "data.csv")

        # Create dummy manuscript
        with open(self.manuscript_path, "w") as f:
            f.write("This is a test manuscript about material science and battery lifetime prediction.")

        # Create dummy dataset
        pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}).to_csv(self.data_path, index=False)

        # Init AutoFeaturizer
        self.state = AutoFeaturizer(
            target="test_target",
            manuscript_path=self.manuscript_path,
            data_path=self.data_path
        )

    def test_summarizer_real_llm(self):
        summarizer_agent = summarize(dummy_llm, max_retries=2)
        summarizer_agent(self.state)

        # Assertions
        self.assertIsNotNone(self.state.literature_review)
        print("\n=== Manuscript Summary ===")
        print(self.state.literature_review)

        self.assertIsInstance(self.state.features_description, dict)
        print("\n=== Feature Descriptions ===")
        for k, v in self.state.features_description.items():
            print(f"{k}: {v}")

        self.assertTrue(len(self.state.features_description) > 0)


if __name__ == "__main__":
    unittest.main()
