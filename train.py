import os
import time
import numpy as np
import pandas as pd

from utils import action_to_order, load_data
from trading_environment import TradingEnv

data_df = load_data("data/Bitfinex_BTCUSD_1h_2019")

EPISODES = 50_000

WINDOW_SIZE = 50
INITIALIZE_BALANCE = 100_000
POSITION_SIZE = 1000 # 1% of 100_000 is 1000



env = TradingEnv(data_df, balance=INITIALIZE_BALANCE, window_size=WINDOW_SIZE)

total_time = 0
for episode in range(EPISODES):
    episode_start_time = time.time()
    print(f"----------- Episode:{episode+1}/{EPISODES} -----------")
    env.reset()
    done = False
    state = env.get_observation()

    while not done:
        
        action = 0
        
        orders = action_to_order(action)
        state, reward, done = env.step(orders)
    
    episode_end_time = time.time()
    total_time += episode_end_time - episode_start_time
    # print("--- %s seconds ---" % (episode_end_time - episode_start_time))

print(f"total time: {total_time}")