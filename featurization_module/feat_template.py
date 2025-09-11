import pandas as pd

def generate_new_feats(data: pd.DataFrame) -> pd.DataFrame:
    data.loc[:, 'A+B'] = data['A'] + data['B']

    return data