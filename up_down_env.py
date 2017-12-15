from gym import spaces
import numpy as np

class UpDownEnv(object):

    def __init__(self, data):
        self.data = data
        self.action_space = spaces.Discrete(3)
        self.current_action = None
        self.i = 1
        self.metadata = {'render.modes': []}
        self.reward_range = (-np.inf, np.inf)

        # data must be numpy array
        self.observation_space = \
        spaces.Box(low=np.min(self.data), high=np.max(self.data), \
            shape=self.data.shape)

    def step(self, action):
        # set up defaults
        observation = self.data[self.i]
        self.i += 1

        reward = 0
        done = False
        info = {}


        # see if there is another observation to take
        try:
            observation = self.data[self.i]
        except:
            done = True
        
        # process the actions (1 is long, 2 is short)
        if not done:
            if self.current_action == 1:
                reward = self.data[self.i][0] - self.data[self.i-1][0]
            if self.current_action == 2:
                reward = self.data[self.i-1][0] - self.data[self.i][0]

        # set the new action and return
        reward *= 10000
        self.current_action = action
        return observation, reward, done, info

    def reset(self):
        self.i = 1
        return self.data[self.i]

    # def render(self, mode='human', close=False):
    #     """Renders the environment.
    #     The set of supported modes varies per environment. (And some
    #     environments do not support rendering at all.) By convention,
    #     if mode is:
    #     - human: render to the current display or terminal and
    #       return nothing. Usually for human consumption.
    #     - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
    #       representing RGB values for an x-by-y pixel image, suitable
    #       for turning into a video.
    #     - ansi: Return a string (str) or StringIO.StringIO containing a
    #       terminal-style text representation. The text can include newlines
    #       and ANSI escape sequences (e.g. for colors).
    #     Note:
    #         Make sure that your class's metadata 'render.modes' key includes
    #           the list of supported modes. It's recommended to call super()
    #           in implementations to use the functionality of this method.
    #     Args:
    #         mode (str): the mode to render with
    #         close (bool): close all open renderings
    #     Example:
    #     class MyEnv(Env):
    #         metadata = {'render.modes': ['human', 'rgb_array']}
    #         def render(self, mode='human'):
    #             if mode == 'rgb_array':
    #                 return np.array(...) # return RGB frame suitable for video
    #             elif mode is 'human':
    #                 ... # pop up a window and render
    #             else:
    #                 super(MyEnv, self).render(mode=mode) # just raise an exception
    #     """
    #     if not close: # then we have to check rendering mode
    #         modes = self.metadata.get('render.modes', [])
    #         if len(modes) == 0:
    #             raise error.UnsupportedMode('{} does not support rendering (requested mode: {})'.format(self, mode))
    #         elif mode not in modes:
    #             raise error.UnsupportedMode('Unsupported rendering mode: {}. (Supported modes for {}: {})'.format(mode, self, modes))
    #     return self._render(mode=mode, close=close)
