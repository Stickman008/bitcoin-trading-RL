from collections import deque
import random

import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop

class A2C():
    
    def __init__(self):
        pass
    
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
                
        return model
    
    def act(self, state):
        pass
    
    def loss(self):
        pass
    
    def save(self, path):
        pass
