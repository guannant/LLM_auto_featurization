"""
Builds the LangGraph pipeline for automatic featurization with feedback loop.
"""

from langgraph.graph import StateGraph, END

# Import agents
from auto_feat.first_pass.summarization.summarize import summarize
from auto_feat.featurization_module.proposal import feat_proposal
from auto_feat.featurization_module.execution import feature_generation
from auto_feat.eval_module.evaluator import create_evaluation_agent_wrap

# Import LLM API wrapper
from auto_feat.LLM_API.LLM_chat import chatbox


def build_autofeat_graph(task: str = "regression", max_retries: int = 5):
    """
    Build the LangGraph pipeline with feedback loop.

    Flow:
        - Summarizer initializes AutoFeaturizer with literature + dataset.
        - Evaluation runs on the original dataset to produce a baseline report.
        - Feedback loop:
            Evaluation → Proposal → Generation → Evaluation
        - Stops after max_iterations.

    Args:
        task (str): "regression" or "classification".
        max_retries (int): retries for LLM-based modules.
    Returns:
        workflow (StateGraph)
    """

    workflow = StateGraph(dict)

    # --- Summarization (initialization only) ---
    summarizer = summarize(chatbox, max_retries=max_retries)
    workflow.add_node("Summarizer", summarizer)

    # --- Proposal agent ---
    proposal_agent = feat_proposal(chatbox, max_retries=max_retries)
    workflow.add_node("FeatProposal", proposal_agent)

    # --- Feature Generation agent ---
    generation_agent = feature_generation(chatbox, max_retries=max_retries)
    workflow.add_node("FeatGeneration", generation_agent)

    # --- Evaluation agent ---
    eval_agent = create_evaluation_agent_wrap(max_retries=max_retries, task=task)
    workflow.add_node("Evaluation", eval_agent)

    # --- Workflow wiring ---
    # Entry: Summarizer → Evaluation (baseline)
    workflow.set_entry_point("Summarizer")
    workflow.add_edge("Summarizer", "Evaluation")

    # --- Conditional feedback loop ---
    def should_continue(state: dict) -> bool:
        """Check iteration count to decide loop continuation."""
        iteration = state.iterations
        max_iter = state.max_iterations
        # Increment iteration in state
        state.iterations = iteration + 1
        return state.iterations <= max_iter

    workflow.add_conditional_edges(
        "Evaluation",
        should_continue,
        {True: "FeatProposal", False: END}
    )

    # Loop body: Proposal → Generation → Evaluation
    workflow.add_edge("FeatProposal", "FeatGeneration")
    workflow.add_edge("FeatGeneration", "Evaluation")

    return workflow
