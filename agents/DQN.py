from collections import deque
import random

import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop


class DQNAgent():
    def __init__(self, state_size, is_eval=False, model_path="model"):
        self.state_size = state_size
        self.action_size = 3  # sit, open_long, close_long
        self.replay_memory = deque(maxlen=1000)
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # min after 919 times

        self.model = load_model(model_path) if is_eval else self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.state_size, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=1e-3))

        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        options = self.model.predict(state)
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.replay_memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path="save_models/test_model.h5"):
        self.model.save(path)


if __name__ == "__main__":
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.997
    cnt = 0
    while epsilon > epsilon_min:
        epsilon *= epsilon_decay
        cnt += 1
    print(cnt)
