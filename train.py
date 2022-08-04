import os
import time
import numpy as np
import pandas as pd

from utils import load_data
from trading_environment import TradingEnv

data_df = load_data("data/Bitfinex_BTCUSD_1h_2019")

EPISODES = 1_000

env = TradingEnv(data_df, balance=10_000, window_size=100)

start_time = time.time()
for episode in range(EPISODES):
    print(f"----------- Episode:{episode}/{EPISODES} -----------")
    env.reset()
    done = False
    state = env.get_observation()

    while not done:
        orders = []
        state, reward, done = env.step(orders)

print("--- %s seconds ---" % (time.time() - start_time))