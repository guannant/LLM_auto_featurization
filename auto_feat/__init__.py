"""
Scripts to perform automatic featurization and model training
"""
from typing import Dict, List, Union

import pandas as pd


class AutoFeaturizer():
    """
    Class that, given a user-specified target, some literature, and relevant data, employs LLMs to automatically
    generates features to be used in downstream ML tasks.

    Args:
        papers: list of paths to where papers are stored (in raw text format)
        data: path to where data is stored
        target: user-specified target to be used by downstream ML models
    """
    def __init__(self,
                 papers: List[str],
                 data: Union[str],
                 target: str) -> None:
        pass

    @property
    def literature_review(self) -> str:
        """
        Returns a relevant summary of the paper(s) provided
        """
        pass

    @property
    def features_description(self) -> Dict:
        """
        Returns a dictionary containing physical descriptions of the features used and created thus far
        """
        pass

    @property
    def clean_augmented_data(self) -> pd.DataFrame:
        """
        Returns dataframe containing original (raw) features, as well as any augmented features that have been already
        used
        """
        pass
    @property
    def construct_strategy(self) -> Dict:
        """
        Returns a dictionary containing proposed (new) features as the key as well as how to construct them as the value
        """
        pass
    