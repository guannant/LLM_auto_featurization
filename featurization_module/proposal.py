from typing import Dict, Tuple

from .utils import PreviousRunsReports

def feat_proposal(description: Dict, summary: str, target: str, report: PreviousRunsReports) -> Tuple[Dict]:
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
    pass