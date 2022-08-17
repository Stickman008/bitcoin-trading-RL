import os
import random
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
    
def action_to_order(action, amount=None, id=None):
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

def random_game(env, episodes, actions=None):
    if actions is None:
        actions = [0, 1, 2] # sit, open_long, close_long
    for episode in range(episodes):
        print(f"----------- Episode:{episode+1}/{episodes} -----------")
        env.reset()
        done = False

        while not done:
            action = random.choice(actions)
            orders = action_to_order(action, amount=None, id=None)
            state, reward, done = env.step(orders)
    