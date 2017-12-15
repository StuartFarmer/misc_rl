import numpy as np
import gym
from gym import spaces

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape, GRU
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import csv

class Env(object):
    """The abstract environment class that is used by all agents. This class has the exact
    same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
    OpenAI Gym implementation, this class only defines the abstract methods without any actual
    implementation.
    """
    def __init__(self):
        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(6)
        self.index = 0

        self.price = []
        with open('daily.csv', 'r') as f:
            reader = csv.reader(f)
            f = list(reader)
            price = f
            for p in price:
                self.price.append(float(p[0]))

        self.diff = []
        with open('daily_diff.csv', 'r') as f:
            reader = csv.reader(f)
            f = list(reader)
            diff = f
            for d in diff:
                self.diff.append(float(d[0]))

        self.in_long = False
        self.in_short = False
        self.change_since_entry = 0
        self.entered_value = 0
    """
    ACTIONS:
    [Pass, Enter_Long, Enter_Short, Exit]

    ENV:
    [Change, Price, In_Long, In_Short, Change_Since_Entry, Entered_Value]
    """
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        reward = 0
        done = False

        # actions
        if action == 1:
            if self.in_long or self.in_short:
                pass
            else:
                entered_value = self.price[self.index]
                self.in_long = True

        if action == 2:
            if self.in_long or self.in_short:
                pass
            else:
                entered_value = self.price[self.index]
                self.in_short = True

        if action == 3:
            if self.in_long:
                reward += self.change_since_entry * 100
                self.entered_value = 0
                self.change_since_entry = 0
                self.in_long = False
            if self.in_short:
                reward += self.change_since_entry * 100
                self.entered_value = 0
                self.change_since_entry = 0
                self.in_short = False

        # update state variables
        if self.in_long:
            self.change_since_entry = pow(self.price[self.index] - self.entered_value, 3)

        if self.in_short:
            self.change_since_entry = pow(-1 * (self.price[self.index] - self.entered_value), 3)

        # step to next value in data
        self.index += 1
        if self.index == len(self.diff) - 1:
            done = True

        observation = [self.diff[self.index], self.price[self.index], 1 if self.in_long else 0, 1 if self.in_short else 0, self.change_since_entry, self.entered_value]
        return observation, reward, done, {}
    def render(self, mode='human', close=False):
        pass
    def reset(self):
        self.index = 0
        self.price = []
        with open('daily.csv', 'r') as f:
            reader = csv.reader(f)
            f = list(reader)
            price = f
            for p in price:
                self.price.append(float(p[0]))

        self.diff = []
        with open('daily_diff.csv', 'r') as f:
            reader = csv.reader(f)
            f = list(reader)
            diff = f
            for d in diff:
                self.diff.append(float(d[0]))

        self.in_long = 0
        self.in_short = 0
        self.change_since_entry = 0
        self.entered_value = 0
        return [self.diff[self.index], self.price[self.index], 1 if self.in_long else 0, 1 if self.in_short else 0, self.change_since_entry, self.entered_value]

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)

# Get the environment and extract the number of actions.
env = Env()
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Reshape(input_shape=(1,6), target_shape=(1, 6)))
model.add(GRU(16, return_sequences=True))
model.add(Activation('relu'))
model.add(GRU(16, return_sequences=True))
model.add(Activation('relu'))
model.add(GRU(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

# After training is done, we save the final weights.
ENV_NAME = 'forex'
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)