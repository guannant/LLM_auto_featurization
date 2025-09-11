def create_your_llm_agent_wrap(llm, max_retries=10):
    #max_retries is used for rerun the llm call if the output is not in the correct format (sometimes llm can output invalid json etc)
    def agent_node_1(state):
        # state: AutoFeaturizer class instance
        input1 = state.property1
        input2 = state.property2

        # ---------------- SYSTEM PROMPT ----------------
        system_message = (
            "System: Define the role for your LLM agent.\n\n"
            "Problem summary:Define your task and objective\n"
            "Output format (STRICT): Define your output format\n"
        )


        # ---------------- USER PROMPT ----------------
        user_msg = (
            "\n==== Input 1 ====\n"
            f"{input1}\n"
            "==== Input 2 ====\n"
            f"{input2}\n"
            "\n\nInstructions:\n"
            "extra instructions for your LLM agent if needed\n"
        )
        prompt = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_msg}
            ]
        # Retry loop:
        # define your own is_valid_result function to check if the result is valid
        def is_valid_result(result):
            # Implement your validation logic here
            return True  # Placeholder, replace with actual validation
        
        for _ in range(max_retries):
            raw = llm(prompt)
            if is_valid_result(raw):
                state.property3 = raw  # update the state with the result
                return
        # If we exhaust all retries, we can return the an error or raise an exception
        raise RuntimeError(f"Failed after {max_retries} retries. Last output: {raw}")

    return agent_node_1
