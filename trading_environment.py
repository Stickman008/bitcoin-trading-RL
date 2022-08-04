from itertools import accumulate
from pyexpat import features
from turtle import position
import numpy as np
import pandas as pd

class TradingEnv():
    
    def __init__(self, df, balance=10_000, window_size=20, nomalize_value=100_000, fees=0.01):
        self.df = df.copy()
        self.window_size = window_size
        self.current_step = window_size
        self.total_steps = len(self.df)-1
        
        self.initalize_balance = balance
        self.balance = balance
        self.total_asset = 0
        self.fees = fees
        self.position_id_increment = 0
        self.positions = dict()
        
        self.accumulate_penalty = 0

        self.nomalize_value = nomalize_value
        self.date, self.prices, self.features = self.process_data()
    

    def process_data(self):
        date = self.df.loc[:, "Date"].to_numpy()
        prices = self.df.loc[:, "Close"].to_numpy()
        features = self.df.drop(["Date", "Open", "High", "Low", "Close"], axis=1).to_numpy()
        return date, prices, features

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initalize_balance
        self.accumulate_penalty = 0
        self.positions = dict()
        self.position_id_increment = 0
    
    # actions sample -> [{"order_type": "open_long", "amount": 1000}, {"order_type": "close_long", "id": 1}]
    # if actions is empty list -> sit
    def step(self, actions):
        self.current_step += 1
        current_price = self.getCurrentPrice()
        current_date = self.getCurrentDate()
        reward = self.get_reward(actions)
        
        for action in actions:
            if action["order_type"] == "open_long" and self.balance >= action["amount"]:
                asset_amount = action["amount"] / current_price
                self.balance -= asset_amount * current_price
                self.total_asset += asset_amount
                self.positions[self.position_id_increment] = {"order_type": "long", "date": current_date,"amount": asset_amount, "entry_price": current_price}
                self.position_id_increment += 1
                
            elif action["order_type"] == "close_long" and action["id"] in self.positions.keys():
                self.total_asset -= self.positions[action["id"]]["amount"]
                self.balance += self.positions[action["id"]]["amount"] * current_price
                del self.positions[action["id"]]
            
            # elif action["order_type"] == "open_short" and self.balance >= action["amount"]:
            #     asset_amount = action["amount"] / current_price
            #     self.balance -= asset_amount * current_price
            #     self.total_asset += asset_amount
            #     self.positions[self.position_id_increment] = {"order_type": "short", "amount": asset_amount, "entry_price": current_price}
            #     self.position_id_increment += 1
            
            # elif action["order_type"] == "close_short" and action["id"] in self.positions.keys():
            #     self.total_asset -= self.positions[action["id"]]["amount"]
            #     self.balance += self.positions[action["id"]]["amount"] * current_price
            #     del self.positions[action["id"]]
            
            # elif action["order_type"] == "close_all_long":
            #     for position in self.positions:
            #         self.total_asset -= self.positions[action["id"]]["amount"]
            #         self.balance += self.positions[action["id"]]["amount"] * current_price
            #         del self.positions[action["id"]]
            else:
                pass
        
        next_obs = self.get_observation()
        done = self.current_step == self.total_steps
        return next_obs, reward, done
    

    def get_observation(self, features=False):
        obs = self.prices[self.current_step-self.window_size: self.current_step].reshape(-1, 1)
        if features:
            # print(obs.shape, self.features[self.current_step-self.window_size: self.current_step, :].shape)
            obs = np.concatenate([obs, self.features[self.current_step-self.window_size: self.current_step, :]], axis=1)
        return obs

    def get_reward(self, actions):
        current_price = self.getCurrentPrice()
        reward = 0
        if len(actions) == 0:
            penalty = self.getBalance()/self.initalize_balance
            self.accumulate_penalty += penalty
        else:
            self.accumulate_penalty = 0 # reset penalty when make action
            for action in actions:
                action_type = action["order_type"]
                if action_type == "open_long" or action_type == "open_short":
                    pass
                else:
                    if action_type == "close_long":
                        reward += (current_price-self.positions[action["id"]]["entry_price"])*self.positions[action["id"]]["amount"]
                        # print(current_price-self.positions[action["id"]]["entry_price"])
                    elif action_type == "close_short":
                        reward += (self.positions[action["id"]]["entry_price"]-current_price)*self.positions[action["id"]]["amount"]

        return reward - self.accumulate_penalty

    def getBalance(self):
        return self.balance
    
    def getNetWorth(self):
        return self.balance + self.total_asset * self.getCurrentPrice()

    def getCurrentPrice(self):
        return self.prices[self.current_step]
    
    def getCurrentDate(self):
        return self.date[self.current_step]

    def getCurrentPostion(self):
        # date, id, order_type, amount(asset), pnl
        current_price = self.getCurrentPrice()

        positions_list = list()
        for position_id, position in self.positions.items():
            if position["order_type"] == "long":
                pnl = (current_price-position["entry_price"]) * position["amount"]
                percent_change = ((current_price-position["entry_price"])/position["entry_price"] )* 100
            elif position["order_type"] == "short":
                pnl = (position["entry_price"]-current_price) * position["amount"]
                percent_change = ((position["entry_price"]-current_price)/position["entry_price"] )* 100

            positions_list.append([position["date"], position_id, position["order_type"], position["entry_price"], position["amount"], pnl, percent_change])

        
        result = pd.DataFrame(positions_list, columns=["Date", "Id", "Type", "EntryPrice", "Amount", "PNL","pChange"])
        return result
    