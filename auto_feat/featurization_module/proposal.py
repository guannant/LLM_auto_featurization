import json

def feat_proposal(llm, max_retries=3):
    """
    Proposes new features to be created from existing features.
    Limits proposals to simple, interpretable features (max 10).
    """
    def agent_node(state):
        description = state.features_description
        summary = state.literature_review
        target = state.target
        report = state.eval_report

        system_message = (
            "You are a scientific feature engineering assistant.\n\n"
            "Task: Propose new features to create from existing features. "
            "Use the given feature descriptions, literature summary, target definition, "
            "and previous run reports.\n\n"
            "STRICT RULES FOR FEATURE CREATION:\n"
            "1. Propose at most 10 new features.\n"
            "2. Each feature must use simple operations:\n"
            "   - basic arithmetic (+, -, *, /)\n"
            "   - ratios or differences\n"
            "   - statistical summaries (mean, variance, min, max, std)\n"
            "3. Each feature can involve at most 3 original columns.\n"
            "4. Use no more than 5 operations per feature.\n"
            "5. Avoid overly complex, nested, or hard-to-compute transformations.\n"
            "6. Prefer simple and interpretable features that are meaningful for science and ML.\n\n"
            "Output format (STRICT JSON Dictionary):\n"
            "{\n"
            '  \"new_feature_computation\": { \"feature_name\": \"explanation of how to derive from existing features\", ... }\n'
            "}\n"
        )

        # Convert report dict into readable string for LLM
        report_str = json.dumps(report, indent=2)

        user_msg = (
            "\n==== Existing Features ====\n"
            f"{description}\n"
            "==== Literature Summary ====\n"
            f"{summary}\n"
            "==== Target Specification ====\n"
            f"{target}\n"
            "==== Previous Runs Report ====\n"
            f"{report_str}\n"
            "\nInstructions:\n"
            "- Suggest no more than 5 simple, interpretable features.\n"
            "- Use only basic arithmetic, ratios, differences, or statistical summaries.\n"
            "- Each feature must be practical to compute in pandas/numpy.\n"
            "- Follow the strict JSON format.\n"
        )

        prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_msg}
        ]

        def is_valid_result(result):
            try:
                parsed = json.loads(result)
                return "new_feature_computation" in parsed
            except Exception:
                return False
    
        raw = None
        for _ in range(max_retries):
            raw = llm(prompt)
            if is_valid_result(raw):
                parsed = json.loads(raw)  
                state.construct_strategy = parsed["new_feature_computation"]
                return 

        raise RuntimeError(f"Failed after {max_retries} retries. Last output: {raw}")

    return agent_node
