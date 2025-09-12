"""
Main entry point for the AutoFeaturizer pipeline.
"""

import os
from auto_feat import AutoFeaturizer
from auto_feat.build_graph import build_autofeat_graph


def main():
    # --- Ensure data directory exists ---
    data_dir = os.path.join("auto_feat", "data")
    os.makedirs(data_dir, exist_ok=True)

    # --- Paths ---
    manuscript_path = os.path.join(data_dir, "manuscript.txt")
    data_path = os.path.join(data_dir, "data.csv")

    # --- Initialize AutoFeaturizer ---
    state = AutoFeaturizer(
        target="OUTPUT PROPERTY: YS (MPa)",
        data_path=data_path,
        manuscript_path=manuscript_path,
    )

    # --- Build LangGraph workflow ---
    workflow = build_autofeat_graph(task="regression", max_retries=10)
    app = workflow.compile()

    # --- Run pipeline ---
    print("🚀 Running AutoFeaturizer pipeline...")
    app.invoke(state)

    # --- Display results ---
    print("\n=== Literature Review ===")
    print(state.literature_review)

    print("\n=== Features Description ===")
    for k, v in state.features_description.items():
        print(f"{k}: {v}")

    print("\n=== Evaluation Report ===")
    print(state.report)

    print("\n=== Construct Strategy ===")
    print(state.construct_strategy)

    print("\n=== Final DataFrame Head ===")
    print(state.clean_augmented_data.head())


if __name__ == "__main__":
    main()
