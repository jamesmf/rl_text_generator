import re
import os
import random
import json
import pickle
from collections import namedtuple
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy
from keras.preprocessing.sequence import pad_sequences
from gym.utils import seeding
from gym import spaces

DEFAULT_CHAR_DICT = {"": 0, "\n": 1, " ": 2, "%": 3, "$": 4, "'": 5, ")": 6,
                     "(": 7, "-": 8, ",": 9, "/": 10, ".": 11, "1": 12,
                     "0": 13, "3": 14, "2": 15, "5": 16, "4": 17, "7": 18,
                     "6": 19, "9": 20, "8": 21, ":": 22, "backspace": 50,
                     "a": 23, "`": 24, "c": 25, "b": 26, "e": 27, "d": 28,
                     "g": 29, "f": 30, "i": 31, "h": 32, "k": 33, "j": 34,
                     "m": 35, "l": 36, "o": 37, "n": 38, "q": 39, "p": 40,
                     "s": 41, "r": 42, "u": 43, "t": 44, "w": 45, "v": 46,
                     "y": 47, "x": 48, "z": 49}

class Environment:

    def __init__(self, evaluators, char_dict, max_steps, log_id,
                 end_seq='$$$$', max_gen_docs=50000):
        """
        Environment object to control rewarding an agent.
        Arguments:
            evaluators (list): list of objects with an evaluate() func to calculate reward
            agent (TextDQNAgent): agent to train
            max_steps (int): maximum steps per episode (document).
        """
        self.evaluators = evaluators
        self.char_dict = char_dict
        self.num_actions = len(char_dict)
        self.max_steps = max_steps
        self.max_len = max_steps
        self.max_gen_docs = max_gen_docs
        self.log_id = log_id

        self.episode_counter = 0
        self.freshly_generated = []
        self.seed_regex = re.compile("name:\n.{4,60}\n")
        self.end_seq = '$$$$'
        max_char = np.max(self.char_dict.values())
        low = np.zeros((self.max_len,))
        high = np.ones((self.max_len,))*max_char

        # gym api for keras-rl
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=low, high=high,
                                            dtype=np.uint16)
        #self.reset()

    def update(self, **kwargs):
        """
        Calls update on all of the Environment's evaluators
        """
        for evaluator in self.evaluators:
            if hasattr(evaluator, 'update'):
                evaluator.update(self, **kwargs)

    def evaluate(self):
        self.reward = 0.
        self.all_rewards = [0. for _ in self.evaluators]
        for n, evaluator in enumerate(self.evaluators):
            r = evaluator.evaluate(self)
            self.reward += r
            self.all_rewards[n] = r
        self.episode_reward += self.reward

    def new_episode(self, training=True):
        self.episode_done = False
        self.episode_reward = 0
        self.step_num = 0
        seed_text, original_doc = self.get_seed(training=training)
        self.original_doc = original_doc
        self.seed_text = seed_text
        self.state_text = seed_text
        self.state_vec = self.text_to_vec(self.state_text)
        self.last_state_text = None
        self.last_state_vec = None
        self.episode_counter += 1

    def text_to_vec(self, text):
        return np.array([self.char_dict[i] for i in text], dtype=np.uint16)

    def take_action(self, action):
        self.last_state_text = self.state_text
        self.last_state_vec = self.state_vec

        if action == "backspace":
            self.backspace()
        else:
            self.state_text += action
            self.state_vec = np.append(self.state_vec, self.char_dict[action])
        self.step_num += 1
        if self.step_num >= self.max_steps:
            self.episode_done = True
        if self.state_text.find(self.end_seq) > -1:
            self.episode_done = True
        if self.episode_done:
            self.freshly_generated.append(self.state_text)
            if len(self.freshly_generated) > self.max_gen_docs:
                self.dump_docs(self.log_id)

    def backspace(self):
        if len(self.state_text) > len(self.seed_text):
            self.state_text = self.state_text[:-1]
            self.state_vec = np.append(0, self.state_vec[:-1])

    def get_seed(self, training=True):
        max_attempts = 10
        if training:
            docs = self.real_docs_train
        else:
            docs = self.real_docs_val
        for i in range(0, max_attempts):
            ind = random.choice(range(0, len(docs)))
            doc = docs[ind]
            seed = '\n'.join(doc.split('\n')[:2])+'\n'
            if re.match(self.seed_regex, seed):
                break
            else:
                print("bad seed: {}".format(seed))
        return seed, doc

    def dump_docs(self, log_id, training=True):
        if training:
            tv = "train"
        else:
            tv = "val"
        doc_dir = "../data/generated/gen_{}/".format(log_id)
        if not os.path.exists(doc_dir):
            os.makedirs(doc_dir)
        with open(doc_dir+tv+'_docs.txt', 'a') as f:
            for doc in self.freshly_generated:
                d = doc+self.end_seq
                d = d[:d.find(self.end_seq)]+self.end_seq
                f.write(d+'\n')
        if training:
            self.gen_docs_train += self.freshly_generated
        else:
            self.gen_docs_val += self.freshly_generated
        self.freshly_generated = []

    def step(self, action):
        """
        Implement the gym api
        """
        self.take_action(action)
        self.evaluate()
        return self.state_vec, self.reward, self.episode_done, {}

    def reset(self):
        self.new_episode()
        return self.state_vec

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass

    def render(self):
        print(self.state_text)