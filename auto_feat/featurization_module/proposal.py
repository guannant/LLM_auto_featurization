from typing import Dict, Tuple
from .utils import PreviousRunsReports
import os
import json
import pandas as pd
from typing import TypedDict, Annotated, Sequence
import operator
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Set your key
os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def feat_proposal(llm, max_retries=3):
    """
    Proposes new features to be created from previously used features, reports on model performance and feature
    relevance, literature summary, and target specification

    Args:
        description: dictionary in which each key is an existing feature, and each value is it physical significance
        summary: summary of the pertaining literature
        target: summary of what we want to targed
        report: report summarizing previous run(s) of the model

    Returns:
        tuple of dictionaries with the same keys, which correspond to new features to be generated:
            first one describes the physical significance of the new features
            second one explains how to obtain them from the existing features
    """
    def agent_node(state):
        description = state.description
        summary = state.summary
        target = state.target
        report = state.report

        system_message = (
            "You are a scientific feature engineering assistant.\n\n"
            "Task: Propose new features to create from existing features. "
            "Use the given feature descriptions, literature summary, target definition, "
            "and previous run reports.\n\n"
            "Output format (STRICT JSON):\n"
            "{\n"
            '  \"new_feature_significance\": { \"feature_name\": \"description of physical meaning\", ... },\n'
            '  \"new_feature_computation\": { \"feature_name\": \"explanation of how to derive from existing features\", ... }\n'
            "}\n"
        )

        user_msg = (
            "\n==== Existing Features ====\n"
            f"{description}\n"
            "==== Literature Summary ====\n"
            f"{summary}\n"
            "==== Target Specification ====\n"
            f"{target}\n"
            "==== Previous Runs Report ====\n"
            f"{report}\n"
            "\n\nInstructions:\n"
            "- Suggest features that are physically meaningful.\n"
            "- Base them on literature and past performance.\n"
            "- Ensure the JSON schema is followed strictly.\n"
        )

        # ✅ Proper LangChain message objects
        prompt = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_msg),
        ]

        def is_valid_result(result):
            try:
                parsed = json.loads(result)
                return (
                    "new_feature_significance" in parsed
                    and "new_feature_computation" in parsed
                )
            except Exception:
                return False

        raw = None
        for _ in range(max_retries):
            raw = llm.invoke(prompt).content  # ✅ .invoke and .content
            if is_valid_result(raw):
                parsed = json.loads(raw)
                return (
                    parsed["new_feature_significance"],
                    parsed["new_feature_computation"],
                )

        raise RuntimeError(f"Failed after {max_retries} retries. Last output: {raw}")

    return agent_node


class DummyState:
    def __init__(self, description, summary, target, report):
        self.description = description
        self.summary = summary
        self.target = target
        self.report = report


if __name__ == "__main__":
    description = {
        "charge_current": "Current applied during charging",
        "voltage": "Cell voltage measured during cycling"
    }
    summary = "Literature suggests normalized charge rates and voltage decay trends are predictive of battery health."
    target = "We want to predict remaining useful life (RUL)."
    report = "Previous models found voltage features important, but struggled with different cell capacities."


    state = DummyState(description, summary, target, report)
    agent = feat_proposal(llm)
    significance, computation = agent(state)

    print("=== Proposed Feature Significance ===")
    print(json.dumps(significance, indent=2))
    print("\n=== Proposed Feature Computation ===")
    print(json.dumps(computation, indent=2))
