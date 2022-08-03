import os
import pandas as pd
import pandas_ta as ta

def load_data(path, strategy=None):
    try:
        data = pd.read_csv(path)
    except Exception:
        data = pd.read_csv(path+'.csv')

    if strategy is not None:
        modify_strategy(data, strategy)

    return data

def modify_strategy(df, strategy):
    df.ta.strategy(strategy)