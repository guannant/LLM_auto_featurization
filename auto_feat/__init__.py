"""
Scripts to perform automatic featurization and model training
"""
from typing import Dict, List, Union, Optional, Any
import pandas as pd
import os


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
                 manuscript_path: str = None,
                 data_path: str = None,
                 max_iterations: int = 5) -> None:
        self.iterations = 0 
        self.max_iterations = max_iterations
        base_dir = os.path.join(os.path.dirname(__file__), "data")
        self.manuscript_path = manuscript_path or os.path.join(base_dir, "manuscript.txt")
        self.data_path = data_path or os.path.join(base_dir, "data.csv")
        self.target = target
        self.data = pd.read_csv(self.data_path)

        # === Pipeline-populated attributes ===

        #From paper summarization
        
        self._literature_review: Optional[str] = None
        self._features_description: Dict[str, str] = {}   # original + engineered
        self._clean_augmented_data: pd.DataFrame = self.data.copy()
        self.cur_feature_keys = [col for col in self._clean_augmented_data.columns if col != self.target]


        # From proposal
        self._construct_strategy: Dict[str, str] = {}
        self.new_feature_computation: Optional[Dict[str, str]] = None
        

        # From generation
        self.error_message: Optional[str] = None

        # From evaluation
        self.eval_report: Optional[Dict[str, Any]] = None
        self.datalog = []
        self.newfeaturelog = []

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
        self.cur_feature_keys = list(strategy.keys())
        self.newfeaturelog.append(strategy)
        self._construct_strategy = strategy
