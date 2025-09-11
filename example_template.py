def create_your_llm_agent_wrap(llm, max_retries=10):
    
    def agent_node_1(state):
        input1 = state["input1"]
        input2 = state["input2"]

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
        # Retry loop
        
        result = llm(prompt)
        return {
            **state,
            "results": result,
        }

    return agent_node_1
