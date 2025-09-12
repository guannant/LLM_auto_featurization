# LLM_hackson/auto_feat/build_graph.py

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledGraph

# === Import agents ===
from auto_feat.first_pass.summarization.summarize import create_summarizer_agent_wrap
from auto_feat.featurization_module.proposal import feat_proposal
from auto_feat.featurization_module.execution import feature_generation
from auto_feat.eval_module.evaluator import create_evaluation_agent_wrap

# === Import state class ===
from auto_feat.state import AutoFeaturizer  # you should place your AutoFeaturizer in auto_feat/state.py


def build_pipeline(llm, max_retries: int = 3, task: str = "regression") -> CompiledGraph:
    """
    Constructs the full LangGraph workflow for AutoFeatSci.

    Args:
        llm: LLM backend (LangChain-compatible, e.g., ChatOpenAI).
        max_retries (int): retry attempts for proposal/generation/evaluation.
        task (str): "regression" or "classification".

    Returns:
        Compiled LangGraph workflow.
    """
    workflow = StateGraph(AutoFeaturizer)

    # === Nodes ===
    summarizer_node = create_summarizer_agent_wrap(llm, max_retries=max_retries)
    proposal_node = feat_proposal(llm, max_retries=max_retries)
    generation_node = feature_generation(llm, max_retries=max_retries)
    evaluation_node = create_evaluation_agent_wrap(max_retries=max_retries, task=task)

    # === Register nodes ===
    workflow.add_node("Summarizer", summarizer_node)
    workflow.add_node("Proposal", proposal_node)
    workflow.add_node("Generation", generation_node)
    workflow.add_node("Evaluation", evaluation_node)

    # === Edges ===
    workflow.set_entry_point("Summarizer")
    workflow.add_edge("Summarizer", "Proposal")
    workflow.add_edge("Proposal", "Generation")
    workflow.add_edge("Generation", "Evaluation")

    # Loop back: evaluation → proposal for iterative refinement
    workflow.add_edge("Evaluation", "Proposal")

    # Finish at Evaluation (you can later add a stopping condition)
    workflow.set_finish_point("Evaluation")

    return workflow.compile()
