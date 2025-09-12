"""
Scripts to perform automatic featurization and model training
"""
from typing import Dict, List, Union, Optional, Any
import pandas as pd


class AutoFeaturizer:
    """
    Class that, given a user-specified target, some literature, and relevant data, employs LLMs to automatically
    generate features to be used in downstream ML tasks.

    Args:
        papers: list of paths to where papers are stored (in raw text format)
        data: path to where data is stored (CSV or parquet supported)
        target: user-specified target to be used by downstream ML models
    """

    def __init__(self,
                 target: str,
                 manuscript_path: str,
                 data_path: str) -> None:
        
        self.manuscript_path = manuscript_path
        self.data_path = data_path
        self.target = target

        
        
        

        # === Pipeline-populated attributes ===

        #From paper summarization
        self.papers = None
        self.data = None
        self._literature_review: Optional[str] = None
        self._features_description: Dict[str, str] = {}   # original + engineered
        self._clean_augmented_data: pd.DataFrame = self.data.copy()


        # From proposal
        self._construct_strategy: Dict[str, str] = {}
        self.new_feature_significance: Optional[Dict[str, str]] = None
        self.new_feature_computation: Optional[Dict[str, str]] = None

        # From generation
        self.error_message: Optional[str] = None

        # From evaluation
        self.report: Optional[Dict[str, Any]] = None

    # === Properties ===
    @property
    def literature_review(self) -> str:
        """Returns a relevant summary of the paper(s) provided"""
        return self._literature_review

    @literature_review.setter
    def literature_review(self, summary: str) -> None:
        self._literature_review = summary

    @property
    def features_description(self) -> Dict:
        """Returns dictionary containing physical descriptions of features (original + created)"""
        return self._features_description

    @features_description.setter
    def features_description(self, desc: Dict) -> None:
        self._features_description = desc

    @property
    def clean_augmented_data(self) -> pd.DataFrame:
        """Returns dataframe containing original and augmented features"""
        return self._clean_augmented_data

    @clean_augmented_data.setter
    def clean_augmented_data(self, df: pd.DataFrame) -> None:
        self._clean_augmented_data = df

    @property
    def construct_strategy(self) -> Dict:
        """Returns dictionary containing proposed (new) features and their construction rules"""
        return self._construct_strategy

    @construct_strategy.setter
    def construct_strategy(self, strategy: Dict) -> None:
        self._construct_strategy = strategy
