from typing import Dict, List

import pandas as pd

def feature_generation(available_feats: List[str], proposed_feats: Dict) -> None:
    """
    Creates a python file which outputs a dataframe with new features. Template found in `feat_template.py`

    Args:
        available_feats: list of features that have already been used, and are readily available to be recombined
        proposed_feats: dictionary in which the keys correspond to new features we wish to generate, and the values
            indicate how to generate them from the available features
    """
    pass