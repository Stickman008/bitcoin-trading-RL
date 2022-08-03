from re import A
from cv2 import WND_PROP_ASPECT_RATIO
import numpy as np
import pandas as pd

class TradingEnv():
    
    def __init__(self, df, balance=10_000, window_size=20, nomalize_value=100_000, fees=0.01):
        self.df = df.copy()
        self.window_size = window_size
        self.total_steps = len(self.df)-1
        
        self.initalize_balance = balance
        self.balance = balance
        self.total_asset = 0
        self.fees = fees
        self.position_id_increment = 0
        self.position = dict()
        
        self.nomalize_value = nomalize_value
    
    
    def reset(self):
        self.balance = self.initial_balance
        self.position_id_increment = 0
    
    # actions sample -> [{"order_type": "open_long", "amount": 1000}, {"order_type": "close_long", "id": 1}]
    # if actions is empty list -> sit
    def step(self, actions):        
        current_price = 1
        
        for action in actions:
            if action["order_type"] == "open_long" and self.balance >= action["amount"]:
                asset_amount = action["amount"] / current_price
                self.balance -= action["amount"]
                self.total_asset += asset_amount
                self.position[self.position_id_increment] = {"amount": asset_amount}
                self.position_id_increment += 1
                
            elif action["order_type"] == "close_long" and action["id"] in self.position.keys():
                self.total_asset -= self.position[action["id"]]
                self.balance += self.position[action["id"]] * current_price
                del self.position[action["id"]]
                
            else:
                pass
        
        
        # return next_obs, reward, done
        
    def get_reward(self):
        reward = 0
        
        return reward
    