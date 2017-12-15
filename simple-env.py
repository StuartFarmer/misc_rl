from up_down_env import UpDownEnv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import talib

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

underlying_data = []

with open('./3m_data.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		underlying_data.append(row)

underlying_data = [float(d[4]) for d in underlying_data[1:]]
underlying_data = np.array(underlying_data)

BBAND_TIMEPERIOD = 50
upperband, middleband, lowerband = talib.BBANDS(underlying_data, timeperiod=BBAND_TIMEPERIOD, nbdevup=1, nbdevdn=1)
jaw = talib.EMA(underlying_data, timeperiod=13) # 8
lips = talib.EMA(underlying_data, timeperiod=5) # 3

JAW_OFFSET = 8
for i in range(JAW_OFFSET):
	jaw = np.insert(jaw, 0, np.nan)
jaw = jaw[JAW_OFFSET:]

LIPS_OFFSET = 3
for i in range(LIPS_OFFSET):
	lips = np.insert(lips, 0, np.nan)
lips = lips[LIPS_OFFSET:]

data = np.array([underlying_data[BBAND_TIMEPERIOD:], \
	upperband[BBAND_TIMEPERIOD:], \
	lowerband[BBAND_TIMEPERIOD:], \
	jaw[BBAND_TIMEPERIOD:], \
	lips[BBAND_TIMEPERIOD:]])

data = np.reshape(data, (data.shape[1], data.shape[0]))
print(data.shape)

env = UpDownEnv(data=data)
nb_actions = env.action_space.n
# Next, we build a very simple model.
model = Sequential()
model.add(LSTM(16, input_shape=(30, 5)))
#model.add(Dense(16, input_shape=(1, 5)))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=30)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10000,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)