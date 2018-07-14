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
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Input, Dense, Conv1D, Embedding, Flatten, GlobalMaxPooling1D
from keras.layers import MaxPooling1D, Add, Dropout, BatchNormalization, Concatenate
from keras.optimizers import Adam
from keras.models import Model, load_model


DLM = "$$$$"


def get_log_id(log_dir):
    """
    Increments the maximum log_ID by one and returns it. Finds log_id based
    on the directory names within the logging directory
    """
    patt = re.compile(r"log_(\d+)")
    ls = os.listdir(log_dir)
    ids = [patt.match(i).groups(0)[0] for i in filter(patt.match, ls)]
    log_ids = [int(i) for i in ids]
    if len(log_ids) == 0:
        log_ids = [0]
    return np.max(log_ids)+1


def get_real_docs(data_dir, train=True):
    split = 'val'
    if train:
        split = 'train'
    with open(data_dir+'{}_docs.txt'.format(split), 'r') as f:
        docs = [i+DLM for i in f.read().lower().strip().split(DLM) if i != '']
        docs = [d[d.find("name"):] for d in docs if d.find("name") > -1]
    return docs


def get_generated_docs(data_dir, log_ids, train=True):
    split = 'val'
    if train:
        split = 'train'
    docs = []
    for log_id in log_ids:
        dirname = data_dir+'gen_{}/'.format(log_id)
        with open(dirname+'{}_docs.txt'.format(split), 'r') as f:
            docs += [i+DLM for i in f.read().lower().strip().split(DLM) if i != '']
#    docs = [d[d.find("name"):] for d in docs if d.find("name") > -1]
    return docs


class TextProcessor:

    def __init__(self, max_seq_len):
        """
        A TextProcessor object helps store information useful for encoding raw
        text and for reversing that operation.
        Arguments:
            max_seq_len (int): max length of text you want to generate
        """
        self.max_seq_len = max_seq_len
        self.char_dict = {}
        self.char_rev = {}

    def process_document(self, doc):
        """
        This demo assumes the documents being passed in are pre-processed, but
        should honor this function in case we can't store documents the same
        way we want to train on them.
        """
        return doc

    def fit(self, docs):
        """
        Takes all documents passed in and defines the legal input characters
        and legal outputs.
        Arguments:
            docs (list<str>): input docs (can be training + dev set)
        """
        chars = set([c for r in docs for c in self.process_document(r)])
        self.char_dict[''] = 0
        for c in chars:
            self.char_dict[c] = len(self.char_dict)
        dict_len = len(self.char_dict)
        self.char_dict["backspace"] = dict_len
        self.char_rev = {v: k for k,v in self.char_dict.items()}

    def vectorize(self, doc, seq_len, start=0, end=None):
        """
        Function for converting text into sequences
        Arguments:
            doc (str): document to convert
            seq_len (int): length of output sequence
            start (int): starting character index
            end (int|None): ending character index (if None then len(doc))
        Returns:
            (ndarray): single sequence of character ints, left-padded
        """
        if end is None:
            end = len(doc)
        s = [[self.char_dict[i] for i in doc[start:end]]]
        return pad_sequences(s, maxlen=seq_len)[0]


class Environment:

    def __init__(self, evaluators, agent, max_steps=None, train_interval=10,
                 mem_interval=25, update_interval=400, warmup=100):
        """
        Environment object to control rewarding an agent.
        Arguments:
            evaluators (list): list of objects with an evaluate() func to calculate reward
            agent (TextDQNAgent): agent to train
            max_steps (int): maximum steps per episode (document). if None, tp.max_seq_len
            train_interval (int): how many episodes before retraining model
            mem_interval (int): how many experiences before we store one in memory
            update_interval (int): how many episodes before we update the evaluators
        """
        self.evaluators = evaluators
        self.agent = agent
        self.text_processor = self.agent.text_processor
        if max_steps:
            self.max_steps = max_steps
        else:
            self.max_steps = self.agent.text_processor.max_seq_len
        self.train_interval = train_interval
        self.mem_interval = mem_interval
        self.update_interval = update_interval
        self.episode_counter = 0
        self.warmup = warmup
        self.freshly_generated = []
        self.seed_regex = re.compile("name:\n.{4,60}\n")

    def update(self, **kwargs):
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
        self.step = 0
        seed_text, original_doc = self.get_seed(training=training)
        self.original_doc = original_doc
        self.seed_text = seed_text
        self.state_text = seed_text
        tp = self.text_processor
        self.state_vec = tp.vectorize(self.state_text, 
                                      seq_len=self.agent.max_len).reshape(1, -1)
        self.last_state_text = None
        self.last_state_vec = None
        self.episode_counter += 1
        if hasattr(self.agent, 'epsilon'):
            if hasattr(self.agent, 'epsilon_decay'):
                self.agent.current_epsilon -= self.agent.epsilon_decay

    def take_action(self, action):
        self.last_state_text = self.state_text
        self.last_state_vec = self.state_vec

        if action == "backspace":
            self.backspace()
        else:
            self.state_text += action
            self.state_vec = np.append(self.state_vec[0][1:],
                                       self.text_processor.char_dict[action])
            self.state_vec = self.state_vec.reshape(1, -1)
        self.step += 1
        if self.step >= self.max_steps:
            self.episode_done = True
        if self.state_text.find(DLM) > -1:
            self.episode_done = True
        if self.episode_done:
            self.freshly_generated.append(self.state_text)


    def backspace(self):
        if len(self.state_text) > len(self.seed_text):
            self.state_text = self.state_text[:-1]
            self.state_vec = np.append(0, self.state_vec[0][:-1]).reshape(1, -1)

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
                d = doc+DLM
                d = d[:d.find(DLM)]+DLM
                f.write(d+'\n')
        if training:
            self.gen_docs_train += self.freshly_generated
        else:
            self.gen_docs_val += self.freshly_generated
        self.freshly_generated = []

    def generate_docs(self, num, training=False):
        episode_count = self.episode_counter
        for i in range(0, num):
            self.new_episode(training=False)
            while not self.episode_done:
                action, ind = self.agent.choose_action(self)
                self.take_action(action)
        self.episode_counter = episode_count


class Policy:
    """
    When we refactor, we won't bank on epsilon-greedy
    """
    def __init__(self, policy_type):
        pass


class Discriminator():
    storage_dir = "../models/discriminators/"
    log_dir = "../logs/discriminators/"

    def __init__(self, name, max_len, reward_only_on_end, processor):
        """
        Discriminators predict whether or not a document was generated by one
        of the previous rounds of trained generators or if it was a part of the
        original input documents. The ability to fool a discriminator is one
        thing we will reward our agents for.
        Arguments:
            name (str): name of the discriminator
            max_len (int): number of characters to consider in a document
            reward_only_on_end (Boolean): whether to reward only at final step
        """
        self.name = name        
        self.max_len = max_len
        self.reward_only_on_end = reward_only_on_end
        self.processor = processor
        # make the per-action rewards smaller than the whole-document rewards
        if reward_only_on_end:
            self.reward_norm = 1
        else:
            self.reward_norm = 2./(processor.max_seq_len)

    def evaluate(self, env):
        done = env.episode_done
        state = env.state_text
        if done and self.reward_only_on_end:
            state_vec = self.processor.vectorize(state, seq_len=self.max_len)
            r = self.model.predict(state_vec.reshape(1, -1))[0][0] - 0.5
#            print("global reward: {}".format(r))
            return r * self.reward_norm
        elif not self.reward_only_on_end:
            state_vec = self.processor.vectorize(state, seq_len=self.max_len)
            cr = self.processor.char_rev
#            print(''.join([cr[i] for i in state_vec]))
            r = self.model.predict(state_vec.reshape(1, -1))[0][0] - 0.5
#            print("local reward: {}".format(r))
            return r * self.reward_norm
        return 0

    def get_model(self, path=None, **kwargs):
        """
        Define or load a model
        """
        if path:
            print("loading from {}".format(path))
            self.model = load_model(path)
            self.max_len = self.model.input_shape[1]
            with open(path+'_map.json', 'r') as f:
                self.processor.char_dict = json.load(f)
            self.processor.char_rev = {v: k for k,v in self.processor.char_dict.items()}
        else:
            tp = self.processor
            filter_size = kwargs.get("filter_size", 256)
            embedding_size = kwargs.get("embedding_size", 16)
            num_blocks = kwargs.get("num_blocks", 1)
            char_inp = Input(shape=(self.max_len,))
            emb = Embedding(len(tp.char_dict), embedding_size)(char_inp)
            layer_in = emb
            for n in range(0, num_blocks):
                conv1 = Conv1D(filter_size, 5, padding="same", dilation_rate=1,
                               activation='relu', name='conv_1a')(layer_in)
                conv2 = Conv1D(filter_size, 5, padding="same", dilation_rate=1,
                               activation='relu', name='conv_1b')(conv1)
                conv2 = BatchNormalization()(conv2)
                conv2 = Dropout(0.25)(conv2)
                conv3 = Conv1D(filter_size, 5, padding="same", dilation_rate=2,
                               activation='relu', name='conv_1c')(conv2)
                conv3 = BatchNormalization()(conv3)
                conv3 = Dropout(0.25)(conv3)
                conv4 = Conv1D(filter_size, 5, padding="same", dilation_rate=4,
                               activation='relu', name='conv_1d')(conv3)
                layer_in = conv4
            gmp = GlobalMaxPooling1D()(conv4)
            out = Dense(1, activation='sigmoid', name=self.name+'_out')(gmp)
            model = Model(char_inp, out)
            opt = Adam(0.001)
            model.compile(opt, 'binary_crossentropy')
            self.model = model

    def update(self, env, epochs=1):
        """
        
        """
        whole_doc = self.reward_only_on_end
        gen_train = env.gen_docs_train
        gen_val = env.gen_docs_val
        real_train = env.real_docs_train
        real_val = env.real_docs_val
        self.fit(self.processor, real_train+gen_train,
                 real_val+gen_val,
                 np.append(np.ones(len(real_train)),
                           np.zeros(len(gen_train))),
                 np.append(np.ones(len(real_val)), 
                           np.zeros(len(gen_val))), num_per=1,
                 epochs=epochs, shuffle=True, whole_doc=whole_doc)
        
    def fit(self, text_processor, train_docs, val_docs, train_labels,
            val_labels, num_per=1, epochs=5, shuffle=False, whole_doc=False):
        """
        Taking in a sample of training documents, some original data and some
        generated (by any historical generator), fit a discriminator that sees
        a sample of max_len characters from the document.
        """
        X_train = np.zeros((len(train_docs)*num_per, self.max_len))
        X_val = np.zeros((len(val_docs)*num_per, self.max_len))
        y_train = np.zeros((len(train_docs)*num_per, 1))
        y_val = np.zeros((len(val_docs)*num_per, 1))

        for n, doc in enumerate(train_docs):
            for i in range(0, num_per):
                n2 = n*num_per + i
                if whole_doc:
                    if len(doc) <= self.max_len:
                        start_ind = 0
                        end_ind = len(doc)
                    else:
                        start_ind = np.random.randint(0, len(doc)-self.max_len)
                        end_ind = start_ind + self.max_len
                else:
                    end_ind = np.random.randint(0, len(doc))
                    start_ind = max(0, end_ind-self.max_len)
                x = text_processor.vectorize(doc, self.max_len,
                                             start=start_ind,
                                             end=end_ind)
                X_train[n2, :] = x
                y_train[n2] = train_labels[n]
        for n, doc in enumerate(val_docs):
            for i in range(0, num_per):
                n2 = n*num_per + i
                if whole_doc:
                    if len(doc) <= self.max_len:
                        start_ind = 0
                        end_ind = len(doc)
                    else:
                        start_ind = np.random.randint(0, len(doc)-self.max_len)
                        end_ind = start_ind + self.max_len
                else:
                    end_ind = np.random.randint(0, len(doc))
                    start_ind = max(0, end_ind-self.max_len)
                x = text_processor.vectorize(doc, self.max_len,
                                             start=start_ind,
                                             end=end_ind)
                X_val[n2, :] = x
                y_val[n2] = val_labels[n]
        callbacks = [
            EarlyStopping(patience=2, monitor='val_loss'),
            ModelCheckpoint(filepath=self.storage_dir+'discrim_'+self.name,
                            verbose=1, save_best_only=True,
                            monitor='val_loss'),
            TensorBoard(log_dir='{}/{}'.format(self.log_dir, self.name))
        ]
        self.model.fit(X_train, y_train, epochs=epochs,
                       validation_data=(X_val, y_val),
                       callbacks=callbacks)
        self.model = load_model(self.storage_dir+'discrim_'+self.name)
        with open(self.storage_dir+'discrim_'+self.name+'_map.json', 'w') as f:
            f.write(json.dumps(self.processor.char_dict))


class KLCalculator:
    storage_dir = "../models/discriminators/"

    def __init__(self, name, reward_only_on_end, cv_args={},
                 lda_args={'n_components': 30}, max_reward=2):
        self.name = name
        self.reward_only_on_end = reward_only_on_end
        self.cv = CountVectorizer(**cv_args)
        self.model = LatentDirichletAllocation(**lda_args)
        self.max_reward = max_reward
        if reward_only_on_end:
            self.reward_norm = 1./self.max_reward
        else:
            self.reward_norm = 2/600./self.max_reward

    def fit(self, env):
        docs = env.real_docs_train
        x = self.cv.fit_transform(docs)
        self.model.fit(x)
        with open(self.storage_dir+self.name, 'wb') as f:
            pickle.dump(self, f)

    def evaluate(self, env):
        if env.episode_done or (not self.reward_only_on_end):
            odoc = [env.original_doc]
            ndoc = [env.state_text]
            x0 = self.model.transform(self.cv.transform(odoc))[0]+1e-8
            x1 = self.model.transform(self.cv.transform(ndoc))[0]+1e-8
            kldiv = entropy(x0, x1)
            reward = max(self.max_reward - kldiv, 0)
            return reward*self.reward_norm
        return 0.

    def update(self, env):
        pass

    def get_model(self, model_path):
        with open(model_path, 'rb') as f:
            obj = pickle.load(f)
        self.model = obj.model
        self.cv = obj.cv
        

Experience = namedtuple('Experience', 
                        'state0, action, reward, state1, terminal1')

class Memory:
    """
    Memory that stores Experiences (namedtuple) in a RingBuffer
    """

    def __init__(self, size):
        self.buffer = RingBuffer(size)

    def sample(self, num):
        """
        Samples (without replacement) num Experiences from the memory buffer
        """
        try:
            r = xrange(0, len(self.buffer))
        except NameError:
            r = range(0, len(self.buffer))
        batch_idxs = random.sample(r, num)
        return [self.buffer[i] for i in batch_idxs]

    def append(self, exp):
        """
        Append an Experience to the buffer
        """
        self.buffer.append(exp)

    def inject_expertise(self, env, num, max_ind=1000000):
        """
        Put some 'expert' or 'ground truth' experiences into memory
        Arguments:
            env (Environment): Environment associated with Memory object
            num (int): number of Experiences to insert from ground truth data
        """
        episode_count = env.episode_counter
        for n in range(0, num):
            # sample a seed
            env.new_episode()
            doc = env.original_doc
            doc_len = min(len(doc), max_ind)
            # get a start and end index
            min_end = doc.find(env.seed_text)+len(env.seed_text)
            end = random.randint(min_end, doc_len-1)
            start = max(end-env.text_processor.max_seq_len, 0)
            # if we sampled the last character, the episode is over
            if end >= len(doc)-1:
                done = True
            else:
                done = False
            state_text = doc[start:end]
            tp = env.text_processor
            env.state_vec = tp.vectorize(state_text, 
                                         seq_len=env.agent.max_len).reshape(1, -1) 
            action = doc[end]
            env.state_text = state_text
            env.done = done
            env.take_action(action)
            env.evaluate()
            e = Experience(env.last_state_vec, action, env.reward,
                           env.state_vec, env.episode_done)
            env.memory.append(e)
            # if we generated a document, pop it            
            if env.episode_done:
                env.freshly_generated.pop(-1)
        env.episode_counter = episode_count


class RingBuffer(object):
    """
    RingBuffer implementation borrowed from keras-rl
    """
    def __init__(self, max_len):
        self.max_len = max_len
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(max_len)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Return element of buffer at specific index
        # Argument
            idx (int): Index wanted
        # Returns
            The element of buffer at given index
        """
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.max_len]

    def append(self, v):
        """Append an element to the buffer
        # Argument
            v (object): Element to append
        """
        if self.length < self.max_len:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.max_len:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.max_len
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.max_len] = v