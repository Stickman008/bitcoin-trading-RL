from collections import deque
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop

# class A2CModel():
#     def __init__(self, input_size, action_size):
        

class A2CAgent():
    
    def __init__(self, state_size, is_eval=False, model_path="model"):
        self.state_size = state_size
        self.action_size = 3 # sit, open_long, close_long
        self.is_eval = is_eval
        
        self.model = load_model(model_path) if is_eval else self.create_model()
    
    def create_model(self, custom_model=None):
        model = None
        if custom_model:
            model = clone_model(custom_model)
        else:
            model = Sequential()
            model.add(Dense(512, input_dim=self.state_size, activation="relu"))
            model.add(Dense(256, activation="relu"))
            model.add(Dense(64, activation="relu"))
            model.add(Dense(self.action_size, activation="linear"))
            
        model.compile(loss="mse", optimizer=Adam(learning_rate=1e-3))
        return model
    
    def act(self, state):
        return
    
    def loss(self):
        pass
    
    def save(self, path):
        pass