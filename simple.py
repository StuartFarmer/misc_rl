from up_down_env import UpDownEnv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import talib

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
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