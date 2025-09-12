import time
import openai
from openai import APIStatusError, InternalServerError

# === Model & Client Setup ===
MODEL = "argo:gpt-5"
client = openai.OpenAI(
    api_key="whatever+random",     # Replace with your real key if needed
    base_url="http://0.0.0.0:60963/v1",  # Local server / proxy endpoint
)


def chatbox(prompt, model: str = MODEL, temperature: float = 0.3, max_attempts: int = 5) -> str:
    """
    LLM wrapper around OpenAI Chat API with retry logic.

    Args:
        prompt (list[dict]): Messages in OpenAI chat format [{"role": "system", "content": ...}, ...].
        model (str): Model name to use.
        temperature (float): Sampling temperature.
        max_attempts (int): Maximum retries for transient errors.

    Returns:
        str: LLM response content (string).
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
            # Retry only on 5xx server errors
            if getattr(e, "status_code", 500) >= 500:
                time.sleep(0.5 * (2 ** attempt))  # Exponential backoff
                continue
            raise

        except Exception as e:
            # Retry on transient errors like MIME issues
            if "unexpected mimetype" in str(e).lower():
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise

    raise RuntimeError("ChatCompletion failed after retries")
