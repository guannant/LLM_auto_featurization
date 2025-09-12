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
        max_iterations=3,
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
    for i in range(len(state.datalog)):
        print(f"\n--- Iteration {i+1} ---")
        report = state.datalog[i]
        print(f"Model Type: {report['model_type']}")
        print("Performance (Train):")
        for metric, value in report["performance"]["train"].items():
            print(f"  {metric}: {value}")
        print("Performance (Test):")
        for metric, value in report["performance"]["test"].items():
            print(f"  {metric}: {value}")
        print("Top Features:")
        for feat in report["feature_importance"][:5]:
            print(f"  {feat['variable']}: {feat['relative_importance']:.4f} ({feat['percentage']*100:.2f}%)")
        print(f"Narrative: {report['narrative']}")

    print("\n=== Construct Strategy ===")
    for i in range(len(state.newfeaturelog)):
        print(f"\n--- Iteration {i+1} ---")
        strategy = state.newfeaturelog[i]
        for feat, desc in strategy.items():
            print(f"{feat}:/n")
            print(f"{desc}")

    print("\n=== Final DataFrame Head ===")
    print(state.clean_augmented_data.head())


if __name__ == "__main__":
    main()
