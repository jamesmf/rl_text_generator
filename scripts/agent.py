import re
import os
import random
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Input, Dense, Conv1D, Embedding, Flatten, GlobalMaxPooling1D
from keras.layers import MaxPooling1D, Add, Dropout, BatchNormalization, Multiply
from keras.optimizers import Adam
from keras.models import Model, load_model


class TextDQNAgent:
    """
    Double Deep Q Learning Agent for text generation. Actions are the valid
    characters presented in the training data (from text_processor.fit()).
    State is the text generated thus far. Reward comes from an environment's
    Evaluators. In the simplest case this is a single Discriminator which
    learns to recognize real documents from fake ones.
    """
    model_dir = "../models/dqns/"

    def __init__(self, policy, text_processor, max_state_len,
                 model_path=None, gamma=0.95, epsilon=0.1,
                 update_interval=30, **kwargs):
        self.policy = policy
        self.text_processor = text_processor
        self.actions = [k for k in text_processor.char_dict.keys()]
        self.num_actions = len(self.actions)
        self.max_len = max_state_len
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.000005
        self.current_epsilon = epsilon
        self.update_interval = update_interval
        if model_path:
            self.model = load_model(model_path)
            self.target_model = load_model(model_path)
        else:
            self.model = self.get_model(**kwargs)
            self.target_model = self.get_model(**kwargs)

    def get_model(self, **kwargs):
        tp = self.text_processor
        filter_size = kwargs.get("filter_size", 64)
        embedding_size = kwargs.get("embedding_size", 16)
        num_blocks = kwargs.get("num_blocks", 2)
        categorical = kwargs.get("categorical", False)
        char_inp = Input(shape=(self.max_len,))
        mask = Input(shape=(len(self.actions),))
        emb = Embedding(len(tp.char_dict), embedding_size)(char_inp)
        layer_in = emb
        for n in range(0, num_blocks):
            m = 'conv_{}{}'
            conv1 = Conv1D(filter_size, 5, padding="same", dilation_rate=1,
                           activation='tanh',
                           name=m.format(n, 'a'))(layer_in)
            conv2 = Conv1D(filter_size, 5, padding="same", dilation_rate=1,
                           activation='tanh',
                           name=m.format(n, 'b'))(conv1)
            conv2 = BatchNormalization()(conv2)
            conv2 = Dropout(0.25)(conv2)
            conv3 = Conv1D(filter_size, 5, padding="same", dilation_rate=2,
                           activation='tanh',
                           name=m.format(n, 'c'))(conv2)
            conv3 = BatchNormalization()(conv3)
            conv3 = Dropout(0.25)(conv3)
            conv4 = Conv1D(filter_size, 5, padding="same", dilation_rate=4,
                           activation='tanh',
                           name=m.format(n, 'd'))(conv3)
            pool = MaxPooling1D(pool_size=5)(conv4)
            layer_in = pool
        flat = Flatten()(conv4)
        if categorical:
            out = Dense(len(self.actions), activation='softmax',
                        name='predicted_q')(flat)
        else:
            out = Dense(len(self.actions), activation='relu',
                        name='predicted_q')(flat)
        # this layer masks the output so that we only get gradient from (a)
        m = Multiply()([mask, out])
        model = Model([char_inp, mask], m)
        opt = Adam(0.001)
        if categorical:
            model.compile(opt, 'categorical_crossentropy')
        else:
            model.compile(opt, 'mse')
        return model

    def greedy_init(self, env, model, log_id, epochs=1):
        num_per = 10
        tp = self.text_processor
        docs_train = env.real_docs_train
        docs_val = env.real_docs_val
        X_train = np.zeros((len(docs_train)*num_per, self.max_len))
        y_train = np.zeros((len(docs_train)*num_per, self.num_actions))
        X_val = np.zeros((len(docs_val)*num_per, self.max_len))
        y_val = np.zeros((len(docs_val)*num_per, self.num_actions))
        masks_train = np.ones((len(docs_train)*num_per, self.num_actions))
        masks_val = np.ones((len(docs_val)*num_per, self.num_actions))

        for n, doc in enumerate(docs_train):
            dl = len(doc)
            for n2 in range(0, num_per):
                ind = n*num_per + n2
                end = random.randint(1, dl-1)
                start = max(0, end-self.max_len)
                snippet = doc[start:end]
                letter = doc[end]
                action = tp.char_dict[letter]
                vec = tp.vectorize(snippet, seq_len=self.max_len)
                X_train[ind, :] = vec
                y_train[ind, action] = 1
        for n, doc in enumerate(docs_val):
            dl = len(doc)
            for n2 in range(0, num_per):
                ind = n*num_per + n2
                end = random.randint(1, dl-1)
                start = max(0, end-self.max_len)
                snippet = doc[start:end]
                letter = doc[end]
                action = tp.char_dict[letter]
                vec = tp.vectorize(snippet, seq_len=self.max_len)
                X_val[ind, :] = vec
                y_val[ind, action] = 1
        callbacks = [
            EarlyStopping(patience=6, monitor='val_loss'),
            ModelCheckpoint(filepath=self.model_dir+'init_'+str(log_id),
                            verbose=1, save_best_only=True,
                            monitor='val_loss')
        ]
        self.X_train = X_train
        self.y_train = y_train

        model.fit([X_train, masks_train], y_train, epochs=epochs,
                       validation_data=([X_val, masks_val], y_val),
                       callbacks=callbacks)
        model = load_model(self.model_dir+'init_'+str(log_id))
        return model


    def choose_action(self, env, override_action=None):
        """
        Given the current environment, select the highest value action
        according to the agent's policy (epsilon-greedy for now)
        """
        if override_action:
            return override_action, self.text_processor.char_rev[override_action]
        rv = np.random.rand()
        if rv < self.get_epsilon():
            action_ind = random.choice(range(0, self.num_actions))
            action = self.text_processor.char_rev[action_ind]
            return action, action_ind
        if env.state_vec is None:
            tp = self.text_processor
            env.state_vec = tp.vectorize(env.state_text, seq_len=self.max_len)
        mask = np.ones((1, self.num_actions))
        q_vals = self.model.predict([env.state_vec, mask])[0]
        action_ind = np.argmax(q_vals)
        action = self.text_processor.char_rev[action_ind]
        return action, action_ind

    def train(self, env, batch_size=256):
        """
        Train the online model to better predict Q values. We must first have
        env.warmup Experiences in env.memory.
        """
        experiences = env.memory.sample(batch_size)
        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0[0])
            state1_batch.append(e.state1[0])
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)
        
        state0_batch = np.array(state0_batch)
        state1_batch = [np.array(state1_batch),
                        np.ones((state0_batch.shape[0], self.num_actions))]
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        q_values = self.model.predict_on_batch(state1_batch)
        chosen_actions = np.argmax(q_values, axis=1)
        target_qs = self.target_model.predict_on_batch(state1_batch)
        q_batch = target_qs[range(0, batch_size), chosen_actions]
        
        targets = np.zeros((batch_size, self.num_actions))
        masks = np.zeros((batch_size, self.num_actions))
        discounted_reward_batch = self.gamma * q_batch
        discounted_reward_batch *= terminal1_batch
        Rs = reward_batch + discounted_reward_batch
        for idx, (target, mask, R, action_txt) in enumerate(zip(targets, masks, Rs, action_batch)):
            action = env.text_processor.char_dict[action_txt]
            target[action] = R
            mask[action] = 1.
        #print(state0_batch)
        #print(masks)
        #print(targets)
        self.state0_batch = state0_batch
        self.state1_batch = state1_batch
        self.q_values = q_values
        self.target_qs = target_qs
        self.disc_rew_batch = discounted_reward_batch
        self.Rs = Rs
        self.targets = targets
        val = self.model.train_on_batch([state0_batch, masks], targets)
        #print("training loss: {}".format(val))
        return val

    def save_model(self, log_id):
        model_dir = self.model_dir+"dqn_{}/".format(log_id)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model.save(model_dir+"tdqn_model")

    def get_epsilon(self):
        return max((0, self.current_epsilon))