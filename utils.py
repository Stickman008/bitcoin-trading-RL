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
    df.ta.strategy(strategy, timed=True)
    
def action_to_order(action, amount=1_000, id=0):
    if action == 0:
        return []
    elif action == 1:
        return [{"order_type": "open_long", "amount": amount}]
    elif action == 2:
        return [{"order_type": "close_long", "id": id}]
    elif action == 3:
        return [{"order_type": "open_short", "amount": amount}]
    elif action == 4:
        return [{"order_type": "close_short", "id": id}]
