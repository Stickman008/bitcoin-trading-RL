import os
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import action_to_order, load_data
from trading_environment import TradingEnv
from agents.DQN import DQNAgent
print(tf.config.list_physical_devices('GPU'))

data_df = load_data("data/Bitfinex_BTCUSD_1h_2019")

EPISODES = 1_000
BUFFER_SIZE = 50

WINDOW_SIZE = 50
INITIALIZE_BALANCE = 100_000
POSITION_SIZE = 1000 # 1% of 100_000 is 1000



env = TradingEnv(data_df, balance=INITIALIZE_BALANCE, window_size=WINDOW_SIZE)
agent = DQNAgent(WINDOW_SIZE)

total_time = 0
for episode in range(EPISODES):
    episode_start_time = time.time()
    print(f"----------- Episode:{episode+1}/{EPISODES} -----------")
    env.reset()
    done, cnt_idx = False, 0
    state = env.get_observation()

    while not done:
        action = agent.act(state)
        
        orders = action_to_order(action)
        next_state, reward, done = env.step(orders)

        agent.replay_memory.append((state, action, reward, next_state, done))
        state = next_state

        if len(agent.replay_memory) > BUFFER_SIZE:
            agent.expReplay(WINDOW_SIZE)

        print(cnt_idx)
        cnt_idx += 1
        
    
    episode_end_time = time.time()
    total_time += episode_end_time - episode_start_time

    print(f"Episode: {episode} -> {episode_end_time-episode_start_time} second")
    if episode % 5 == 0 or episode == EPISODES-1:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        agent.save(f"output_models/model_ws_{WINDOW_SIZE}_ep_{episode}_at_{now}")

print(f"total time: {total_time}")
# for 50_000 -> 1636.39 sec = 27.27 min -> 30.56 it/sec (when note perform replay experience)