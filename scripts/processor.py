from rl.core import Processor
import numpy as np


class TextProcessor(Processor):

    def __init__(self, char_dict, max_len):
        self.char_dict = char_dict
        self.char_rev = {v: k for k,v in self.char_dict.items()}
        self.max_len = max_len

    def process_action(self, action):
        if type(action) == tuple:
            # assume action, action_ind
            return action[0]
        elif type(action) == str:
            return action
        else:
            try:
                return self.char_rev[action]
            except KeyError:
                return action

    def process_observation(self, observation, len_override=None):
        ml = self.max_len
        if len_override:
            ml = len_override
        z = np.zeros((ml,))
        x = np.array(observation[-ml:])
        l2 = x.shape[0]
        z[-l2:] = x
        if len_override:
            z = z.reshape(1, -1)
        return z

    def process_state_batch(self, batch):
        """
        TODO: figure out why this is necessary
        """
        batch = np.array(batch)
        if len(batch.shape) == 3:
            batch = batch[0]
        return batch