import re
import os
import random
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Input, Dense, Conv1D, Embedding, Flatten, GlobalMaxPooling1D
from keras.layers import MaxPooling1D, Dropout, BatchNormalization, Multiply
from keras.layers import Lambda, Layer
from keras.optimizers import Adam
from keras.models import Model, load_model
import keras.backend as K


"""
This section directly from keras-rl
"""
def huber_loss(y_true, y_pred, clip_value):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
    elif K.backend() == 'theano':
        from theano import tensor as T
        return T.switch(condition, squared_loss, linear_loss)
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))


def clipped_masked_error(args):
    y_true, y_pred, mask = args
    # this line edited from keras-rl, should pass in delta_clip from self
    loss = huber_loss(y_true, y_pred, np.inf)
    loss *= mask  # apply element-wise mask
    return K.sum(loss, axis=-1)


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

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
                 model_path=None, gamma=0.5, epsilon=0.1,
                 update_interval=50, **kwargs):
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
        char_inp = Input(shape=(self.max_len,))
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
        out = Dense(len(self.actions),
                    name='predicted_q')(flat)
        m = Model(char_inp, out)
        m.compile('sgd', 'mse')

        y_pred = m.output
        y_true = Input(name='y_true', shape=(self.num_actions,))
        mask = Input(name='mask', shape=(self.num_actions,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_true, y_pred, mask])
        ins = [m.input] if type(m.input) is not list else m.input
        trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: [mean_q]}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        opt = Adam(0.001)
        trainable_model.compile(optimizer=opt, loss=losses,
                                metrics=combined_metrics)
        return trainable_model

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
        q_vals = self.model.predict([env.state_vec, mask, mask])[1][0]
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
        ones_mask = np.ones((state0_batch.shape[0], self.num_actions))
        state1_batch = [np.array(state1_batch),
                        ones_mask, ones_mask]
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        q_values = self.model.predict_on_batch(state1_batch)[1]
        chosen_actions = np.argmax(q_values, axis=1)
        target_qs = self.target_model.predict_on_batch(state1_batch)[1]
        q_batch = target_qs[range(0, batch_size), chosen_actions]
        
        targets = np.zeros((batch_size, self.num_actions))
        dummy_targets = np.zeros((batch_size,))
        masks = np.zeros((batch_size, self.num_actions))
        discounted_reward_batch = self.gamma * q_batch
        discounted_reward_batch *= terminal1_batch
        Rs = reward_batch + discounted_reward_batch
        for idx, (target, mask, R, action_txt) in enumerate(zip(targets, masks, Rs, action_batch)):
            action = env.text_processor.char_dict[action_txt]
            target[action] = R
            dummy_targets[idx] = R
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
        self.masks = masks
        self.dummy_targets = dummy_targets
        val = self.model.train_on_batch([state0_batch, targets, masks],
                                        [dummy_targets, targets])
        #print("training loss: {}".format(val))
        return val

    def save_model(self, log_id):
        model_dir = self.model_dir+"dqn_{}/".format(log_id)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model.save(model_dir+"tdqn_model")

    def get_epsilon(self):
        return max((0, self.current_epsilon))