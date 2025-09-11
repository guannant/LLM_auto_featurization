import openai

MODEL = "argo:gpt-4o"

client = openai.OpenAI(
    api_key="whatever+random",
    base_url="http://0.0.0.0:56875/v1",
)


def chat_test():
    print("Running Chat Test with Messages")

    messages = [
        {
            "role": "user",
            "content": "What are you and what is your role?",
        },
    ]
    # max_tokens = 5

    try:

        print(messages)

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            # max_tokens=max_tokens,
        )
        print("Response Body:")
        print(response)
    except Exception as e:
        print("\nError:", e)


if __name__ == "__main__":
    chat_test()
