"""
Define a super class of agent who can implement reinforcement learning
"""
# import modules
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
#Sequential, Dense, Flatten, BatchNormalization, LSTM, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
import numpy as np
import pandas as pd
import random

class RL_Agent(object):
    def __init__(self,*args):   # initialize the learning agent
        pass

    def predict_value(self,*args):  # predict the value function
        pass

    def choose_action(self,*args):  # choose the action
        pass

    def learn(self,*args):   # update the learning agent
        pass

class RL_Net(RL_Agent):   # a network based learner
    def __init__(self, *args):
        pass

    def _build_net(self):
        pass

class ActorNet(RL_Net):
    def __init__(self, state_dim, n_action, n_hidden=20,
                trainer=tf.train.AdamOptimizer(0.001)):
        self.state_dim = state_dim
        self.n_action  = n_action
        self.n_hidden  = n_hidden
        self.trainer   = trainer

        with tf.name_scope('Actor'):
            self._build_net()

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.state_in = tf.placeholder(tf.float32, [None, self.state_dim], name="state_input")
            self.action   = tf.placeholder(tf.int32, [None, ], name="action")
            self.td_error = tf.placeholder(tf.float32, [None, ], name="TD_error")

        # hidden_layer
        with tf.name_scope('hidden_layer'):
            hidden_value = layers.Dense(self.n_hidden, activation='relu')(self.state_in)
        # output_layer
        with tf.name_scope('output_layer'):
            self.action_prob = layers.Dense(self.n_action, activation='softmax')(hidden_value) # use softmax to convert to probability

        with tf.name_scope('logp_grad'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.action)   # this is negative log of chosen action
            # or in this way:
            log_prob = tf.reduce_sum(tf.log(self.action_prob)*tf.one_hot(self.action, self.n_action), axis=1)
            self.logp_grad = tf.reduce_mean(log_prob * self.td_error)  # reward guided loss

        with tf.name_scope('train'):
            self._train_op = self.trainer.minimize(self.logp_grad)

    def choose_action(self, sess, state):
        prob_weights = sess.run(self.action_prob, feed_dict={self.state_in: state[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def learn(self, sess, s, a, td):
        # train on step
        feed_dict = {self.state_in: s, self.action: a, self.td_error: td}
        _, logp_grad = sess.run([self._train_op, self.logp_grad], feed_dict=feed_dict)
        return logp_grad

class CriticNet(RL_Net):
    def __init__(self, state_dim, n_hidden=20, reward_decay=0.9,
                trainer=tf.train.AdamOptimizer(0.01)):
        self.state_dim = state_dim
        self.n_hidden  = n_hidden
        self.gamma     = reward_decay
        self.trainer   = trainer

        with tf.name_scope('Critic'):
            self._build_net()

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.state_in = tf.placeholder(tf.float32, [None, self.state_dim], name="state_input")
            self.v_next   = tf.placeholder(tf.float32, [None, 1], name="next_value")
            self.reward   = tf.placeholder(tf.float32, [None, ], name="reward")

        # hidden_layer
        with tf.name_scope('hidden_layer'):
            hidden_value = layers.Dense(self.n_hidden, activation='relu')(self.state_in)
        # output_layer
        with tf.name_scope('output_layer'):
            self.value = layers.Dense(1, activation=None)(hidden_value) # use softmax to convert to probability

        with tf.name_scope('td_error'):
            self.td_error = self.reward + self.gamma * self.v_next - self.value
            self.loss = tf.reduce_mean(tf.square( self.td_error ))  # reward guided loss

        with tf.name_scope('train'):
            self._train_op = self.trainer.minimize(self.loss)

    def learn(self, sess, s, r, s_):
        # train on step
        v_ = sess.run(self.value, {self.state_in: s_})

        feed_dict = {self.state_in: s, self.v_next: v_, self.reward: r}
        _, td_error = sess.run([self._train_op, self.td_error], feed_dict=feed_dict)
        return td_error

class DDPG(RL_Net):
    def __init__(self, state_dim, action_dim, action_bound=1, n_hidden_a=[30], n_hidden_c=[30], reward_decay=0.9,
                replace_method=dict(name='soft',tau=0.01),
                double_q=False, prioritized=False, dueling=False,
                actor_trainer=tf.train.AdamOptimizer(0.001),critic_trainer=tf.train.AdamOptimizer(0.001)):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.action_bound=action_bound
        self.n_hidden_a = n_hidden_a
        self.n_hidden_c = n_hidden_c
        self.gamma      = reward_decay
        self.act_train  = actor_trainer
        self.crit_train = critic_trainer
        self.rep_method = replace_method
        # options
        self.double_q   = double_q
        self.prioritized= prioritized
        self.dueling    = dueling

        self._build_net()

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/Evaluation_Net')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/Target_Net')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/Evaluation_Net')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/Target_Net')

        if self.rep_method['name'] == 'hard':
            self.at_replace_counter = 0 # count the number of learning steps
            self.ct_replace_counter = 0 # count the number of learning steps
            self.hard_replace_a = [tf.assign(t, e) for t, e in zip(self.at_params, self.ae_params)]
            self.hard_replace_c = [tf.assign(t, e) for t, e in zip(self.ct_params, self.ce_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.rep_method['tau']) * t + self.rep_method['tau'] * e)
                                 for t, e in zip(self.at_params+self.ct_params, self.ae_params+self.ce_params)]

        with tf.variable_scope('target_q'):
            self.q_target = self.reward + self.gamma * self.qsa_

        with tf.variable_scope('dq_TD'):
            self.abs_td = tf.abs(self.q_target-self.qsa)  # for prioritized

        with tf.variable_scope('TD_error'):
            if self.prioritized:
                self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.qsa)) # / tf.reduce_mean(tf.square(self.action[:,0]))
            else:
                self.loss = tf.losses.mean_squared_error(labels=self.q_target, predictions=self.qsa)   # / tf.reduce_mean(tf.square(self.action[:,0]))

        with tf.variable_scope('Critic_trainer'):
            self._train_critic = self.crit_train.minimize(self.loss, var_list=self.ce_params)

        with tf.variable_scope('Actor_loss'):
            self.a_loss = - tf.reduce_mean(self.qsa)

        with tf.variable_scope('Actor_trainer'):
            self._train_actor = self.act_train.minimize(self.a_loss, var_list=self.ae_params)


    def _build_net(self):
        def _build_a(x, n_hidden, scope, trainable=True):  # build actor network
            with tf.name_scope(scope):
                with tf.name_scope('hidden_layers'):
                    hidden_value = layers.Dense(n_hidden[0], activation='relu', trainable=trainable)(x)
                    if len(n_hidden)>1:
                        for i in np.arange(1,len(n_hidden)):
                            hidden_value = layers.Dense(n_hidden[i], activation='relu', trainable=trainable)(hidden_value)
                with tf.name_scope('portf_output_layer'):
                    if self.dueling:
                        # Dueling DQN
                        with tf.name_scope('Value'):
                            self.V = layers.Dense(1, activation=None, trainable=trainable)(hidden_value)

                        with tf.variable_scope('Advantage'):
                            self.A = layers.Dense(self.action_dim, activation=None, trainable=trainable)(hidden_value)

                        with tf.variable_scope('Q'):
                            out = tf.nn.softmax(self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True)))     # Q = V(s) + A(s,a)
                    else:
                        out = layers.Dense(self.action_dim, activation=None, trainable=trainable)(hidden_value)
                with tf.name_scope('hold_output_layer'):
                    hold_out = layers.Dense(1, activation='sigmoid', trainable=trainable)(hidden_value)
            scaled_out = tf.multiply(tf.square(out)/tf.reduce_sum(tf.square(out)), self.action_bound, name='scaled_out')
            return tf.concat([hold_out, scaled_out], 1)

        def _build_c(s, a, n_hidden, scope, trainable=True): # build critic network
            x = tf.concat([s, a], 1)
            with tf.name_scope(scope):
                with tf.name_scope('hidden_layer'):
                    hidden_value = layers.Dense(n_hidden[0], activation='relu', trainable=trainable)(x)
                    if len(n_hidden)>1:
                        for i in np.arange(1,len(n_hidden)):
                            hidden_value = layers.Dense(n_hidden[i], activation='relu', trainable=trainable)(hidden_value)
                with tf.name_scope('output_layer'):
                    out = layers.Dense(1, activation=None, trainable=trainable)(hidden_value)
            return out
        # s inputs
        with tf.name_scope('inputs'):
            self.state_in = tf.placeholder(tf.float32, [None, self.state_dim], name='State_input')
            self.state_ = tf.placeholder(tf.float32, [None, self.state_dim], name='Later_State') # later state for computing target #q
            self.reward = tf.placeholder(tf.float32, [None, 1], name='Rewards')

        with tf.name_scope('Actor'):
            # ---------------- build evaluate_net ------------------
            self.action = _build_a(self.state_in, self.n_hidden_a, 'Evaluation_Net', trainable=True)
            # ---------------- build target_net --------------------
            self.action_= _build_a(self.state_, self.n_hidden_a, 'Target_Net', trainable=False)

        with tf.name_scope('Critic'):
            # ---------------- build evaluate_net ------------------
            self.qsa  = _build_c(self.state_in, self.action, self.n_hidden_c, 'Evaluation_Net', trainable=True)
            # ---------------- build target_net --------------------
            self.qsa_ = _build_c(self.state_, self.action_, self.n_hidden_c, 'Target_Net', trainable=False)

    def choose_action(self, sess, s):
        s = s[np.newaxis, :]    # single state
        return sess.run(self.action, feed_dict={self.state_in: s})[0]  # single action

    def learn(self, sess, s, a, r, s_, ISW=None):   # batch update
        # feed in data
        if self.rep_method['name'] == 'soft':
            sess.run(self.soft_replace)
        else:
            # update actor
            if self.at_replace_counter % self.rep_method['rep_iter_a'] == 0:
                sess.run(self.hard_replace_a)
            self.at_replace_counter += 1
            # update critic
            if self.ct_replace_counter % self.rep_method['rep_iter_c'] == 0:
                sess.run(self.hard_replace_c)
            self.ct_replace_counter += 1

        if self.prioritized:
            feed_dict = {self.state_in: s, self.action: a, self.reward: r, self.state_: s_, self.ISWeights: ISW}

            _, abs_td, loss = sess.run([self._train_critic, self.abs_td, self.loss], feed_dict=feed_dict)
            sess.run(self._train_actor, feed_dict={self.state_in: s})

            return loss, abs_td
        else:
            feed_dict = {self.state_in: s, self.action: a, self.reward: r, self.state_: s_}

            _, loss = sess.run([self._train_critic, self.loss], feed_dict=feed_dict)
            sess.run(self._train_actor, feed_dict={self.state_in: s})

            return loss

class ACNet(RL_Net): # A3C net
    def __init__(self, scope, state_dim, action_dim, action_bound, n_hidden=20, reward_decay=0.9, entropy_beta=0.01,
                actor_trainer=tf.train.AdamOptimizer(0.001), critic_trainer=tf.train.AdamOptimizer(0.002),
                globalAC=None):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.action_bound=action_bound
        self.n_hidden   = n_hidden
        self.beta       = entropy_beta
        self.gamma      = reward_decay
        self.act_train  = actor_trainer
        self.crit_train = critic_trainer

        if ('global' in scope) or ('GLOBAL' in scope) or ('Global' in scope):
            with tf.variable_scope(scope):
                self.state_in = tf.placeholder(tf.float32, [None, self.state_dim], name='state_input')
                self._build_net(scope)
        else:
            with tf.variable_scope(scope):
                self.state_in = tf.placeholder(tf.float32, [None, self.state_dim], name='state_input')
                self.action   = tf.placeholder(tf.float32, [None, self.action_dim], name='action')
                self.v_target = tf.placeholder(tf.float32, [None, 1], name='target_value')

                self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('value_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    self.mu, self.sig = self.mu * self.action_bound, self.sig + 1e-4

                normal_dist = tfp.distributions.Normal(self.mu, self.sig)

                with tf.name_scope('policy_loss'):
                    log_prob = normal_dist.log_prob(self.action)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = self.beta * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_action'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), -self.action_bound, self.action_bound)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('sync'):
                    self.sync_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.sync_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('learn'):
                    self.learn_a_op = self.act_train.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.learn_c_op = self.crit_train.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        with tf.variable_scope('actor'):
            with tf.name_scope('hidden_layer'):
                a_hidden_value = layers.Dense(self.n_hidden, activation='relu')(self.state_in)
            with tf.name_scope('output_layer'):
                self.mu  = layers.Dense(self.action_dim, activation='tanh', name='mu')(a_hidden_value)
                self.sig = layers.Dense(self.action_dim, activation='softplus', name='sigma')(a_hidden_value)
        with tf.variable_scope('critic'):
            with tf.name_scope('hidden_layer'):
                c_hidden_value = layers.Dense(self.n_hidden, activation='relu')(self.state_in)
            with tf.name_scope('output_layer'):
                self.v  = layers.Dense(1, name='value')(c_hidden_value)
        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

    def predict_value(self, sess, s):
        if s.ndim<2: s = s[np.newaxis,:]
        value = sess.run(self.v, {self.state_in: s})[:,0]  # batch prediction predict value
        return value

    def choose_action(self, sess, s):
        action = sess.run(self.A, {self.state_in: s[np.newaxis,:]})  # select action w.r.t the actions prob
        return action

    def learn(self, sess, feed_dict):
        # train on step
        #feed_dict = {self.state_in: s, self.action: a, self.v_target: v}
        _, _, exp_v = sess.run([self.learn_a_op, self.learn_c_op, self.exp_v], feed_dict=feed_dict)
        return exp_v

    def sync_global(self, sess):  # run by a local
        sess.run([self.sync_a_params_op, self.sync_c_params_op])

class AC_RNN(ACNet): # A3C net
    def __init__(self, scope, state_dim, action_dim, action_bound, cell_size=64, n_hidden=50, reward_decay=0.9, entropy_beta=0.01,
                actor_trainer=tf.train.AdamOptimizer(0.001), critic_trainer=tf.train.AdamOptimizer(0.002),
                globalAC=None):
        self.cell_size = cell_size
        super(AC_RNN, self).__init__(scope, state_dim, action_dim, action_bound, n_hidden=n_hidden, reward_decay=reward_decay,
                    entropy_beta=entropy_beta, actor_trainer=actor_trainer, critic_trainer=critic_trainer, globalAC=globalAC)

    def _build_net(self, scope):
        with tf.variable_scope('critic'):
            s = tf.expand_dims(self.state_in, axis=1,
                               name='timely_input')  # [time_step, feature] => [time_step, batch, feature]
            rnn_cell = layers.SimpleRNNCell(self.cell_size)  # tf.contrib.rnn.BasicRNNCell(self.cell_size)  #
            self.init_state = rnn_cell.get_initial_state(batch_size=1, dtype=tf.float32)  # zero_state(batch_size=1, dtype=tf.float32)  #
            outputs, self.final_state = layers.RNN(     # tf.nn.dynamic_rnn
                                    cell=rnn_cell, return_state=True, time_major=True)(s, initial_state=self.init_state)   #
            cell_out = tf.reshape(outputs, [-1, self.cell_size], name='flatten_rnn_outputs')  # joined state representation
            c_hidden = layers.Dense(self.n_hidden, activation='relu', name='hidden_layer')(cell_out)
            self.v = layers.Dense(1, name='output_value')(c_hidden)  # state value

        with tf.variable_scope('actor'):
            with tf.name_scope('hidden_layer'):
                a_hidden_value = layers.Dense(self.n_hidden, activation='relu')(self.state_in)
            with tf.name_scope('output_layer'):
                self.mu  = layers.Dense(self.action_dim, activation='tanh', name='mu')(a_hidden_value)
                self.sig = layers.Dense(self.action_dim, activation='softplus', name='sigma')(a_hidden_value)
        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

    def choose_action(self, sess, s, cell_state):
        action, cell_state = sess.run([self.A, self.final_state], {self.state_in: s[np.newaxis,:], self.init_state: cell_state})  # select action w.r.t the actions prob
        return action, cell_state

class PPO(ACNet): # Proximal Policy Optimization
    def __init__(self, scope, state_dim, action_dim, action_bound, n_hidden=100, reward_decay=0.9, epsilon=0.2,
                actor_trainer=tf.train.AdamOptimizer(0.001), critic_trainer=tf.train.AdamOptimizer(0.002),
                globalAC=None):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.action_bound=action_bound
        self.n_hidden   = n_hidden
        self.epsilon    = epsilon
        self.gamma      = reward_decay
        self.act_train  = actor_trainer
        self.crit_train = critic_trainer
        # super(AC_RNN, self).__init__(scope, state_dim, action_dim, action_bound, n_hidden=n_hidden, reward_decay=reward_decay,
        #             entropy_beta=entropy_beta, actor_trainer=actor_trainer, critic_trainer=critic_trainer, globalAC=globalAC)

        self.state_in = tf.placeholder(tf.float32, [None, self.state_dim], 'state_inputs')
        self.action   = tf.placeholder(tf.float32, [None, self.action_dim], name='action')
        self.v_target = tf.placeholder(tf.float32, [None, 1], 'target_value')
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')

        self._build_net(scope)

        self.td_error = self.v_target - self.v
        self.c_loss = tf.reduce_mean(tf.square(self.td_error))
        self.learn_c_op = self.crit_train.minimize(self.c_loss)

        # choose_action
        self.A = tf.clip_by_value(tf.squeeze(self.a_normal.sample(1), axis=[0,1]), -self.action_bound, self.action_bound)  # operation of choosing action
        self.update_policy = [oldp.assign(p) for p, oldp in zip(self.a_params, self.p_params)]

        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = self.a_normal.prob(self.action) / (self.p_normal.prob(self.action) + 1e-5)
        surr = ratio * self.advantage                       # surrogate loss

        self.a_loss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * self.advantage))
        self.learn_a_op = self.act_train.minimize(self.a_loss)

    def _build_net(self, scope):
        def _build_anet(name, trainable=True):
            with tf.name_scope(name):
                with tf.name_scope('hidden_layer'):
                    a_hidden = layers.Dense(self.n_hidden, activation='relu', trainable=trainable)(self.state_in)
                with tf.name_scope('output_layer'):
                    out_mu  = layers.Dense(self.action_dim, activation='tanh', name='mu', trainable=trainable)(a_hidden)
                    out_sig = layers.Dense(self.action_dim, activation='softplus', name='sigma', trainable=trainable)(a_hidden)
                normal_dist = tfp.distributions.Normal(loc=out_mu, scale=out_sig)
            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/actor/' + name)
            return normal_dist, params

        with tf.variable_scope('critic'):
            with tf.name_scope('hidden_layer'):
                c_hidden = layers.Dense(self.n_hidden, activation='relu', name='hidden_layer')(self.state_in)
            with tf.name_scope('output_layer'):
                self.v = layers.Dense(1, name='output_value')(c_hidden)  # state value
        self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        with tf.variable_scope('actor'):
            self.a_normal, self.a_params = _build_anet('policy', trainable=True)
            self.p_normal, self.p_params = _build_anet('old_policy', trainable=False)
        #self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')

class PolicyGradientDiscrete(RL_Net):  # discrete version of policy Gradient
    def __init__(self, state_dim, n_action, n_hidden=10, reward_decay=0.95,
                trainer=tf.train.AdamOptimizer(0.01)):
        self.n_action  = n_action
        self.state_dim = state_dim
        self.n_hidden  = n_hidden
        self.trainer   = trainer
        self.gamma = reward_decay

        self._build_net()

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.state_in = tf.placeholder(tf.float32, [None, self.state_dim], name="state_input")
            self.action   = tf.placeholder(tf.int32, [None, ], name="action")
            self.qsa      = tf.placeholder(tf.float32, [None, ], name="action_qvalue")

        # hidden_layer
        with tf.name_scope('hidden_layer'):
            hidden_value = layers.Dense(self.n_hidden, activation='relu')(self.state_in)
        # output_layer
        with tf.name_scope('output_layer'):
            self.action_prob = layers.Dense(self.n_action, activation='softmax')(hidden_value) # use softmax to convert to probability

        with tf.name_scope('logp_loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.action)   # this is negative log of chosen action
            # or in this way:
            neg_log_prob = tf.reduce_sum(-tf.log(self.action_prob)*tf.one_hot(self.action, self.n_action), axis=1)
            self.loss = tf.reduce_mean(neg_log_prob * self.qsa)  # reward guided loss

        with tf.name_scope('train'):
            self._train_op = self.trainer.minimize(self.loss)

    def choose_action(self, sess, state):
        prob_weights = sess.run(self.action_prob, feed_dict={self.state_in: state[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def learn(self, sess, s, a, r):
        # discount and normalize episode reward
        qsa = self._discount_and_norm_rewards(r)

        # train on episode
        loss,_ = sess.run([self.loss,self._train_op], feed_dict={
             self.state_in: np.vstack(s),  # shape=[None, state_dim]
             self.action: np.array(a),  # shape=[None, ]
             self.qsa: qsa,  # shape=[None, ]
        })

        return qsa, loss

    def _discount_and_norm_rewards(self, r):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, len(r))):
            running_add = running_add * self.gamma + r[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

class DeepQNetwork(RL_Net):
    def __init__(self, state_dim, n_action, n_hidden=10, reward_decay=0.9,
                e_greedy=0.1, de_greedy=None, e_greedy_min=0.1, replace_target_iter=300,
                trainer=tf.train.AdamOptimizer(0.01), target_net=True,
                double_q=False, prioritized=False, dueling=False):
        #super(RL_DQNet,self).__init__(self,)
        self.state_dim = state_dim
        self.n_action  = n_action
        self.n_hidden  = n_hidden
        self.trainer   = trainer
        self.gamma = reward_decay
        self.eps   = e_greedy
        self.eps_increment = de_greedy
        self.eps_min = e_greedy_min

        self.target_net= target_net
        self.double_q = double_q
        self.prioritized = prioritized
        self.dueling  = dueling
        self.eps = 1 if de_greedy is not None else self.eps_min

        self._build_net()
        self.learn_step_counter  = 0  # count the number of learning steps
        self.replace_target_iter = replace_target_iter
        # self.loss_his = []

        # parameters in target and evaluation networks
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Target_Net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Evaluation_Net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def _build_net(self):
        def _build_model(x, n_hidden, scope):
            with tf.name_scope(scope):
                with tf.name_scope('hidden_layer'):
                    hidden_value = layers.Dense(n_hidden, activation='relu')(x)
                with tf.name_scope('output_layer'):
                    if self.dueling:
                        # Dueling DQN
                        with tf.name_scope('Value'):
                            self.V = layers.Dense(1, activation=None)(hidden_value)

                        with tf.variable_scope('Advantage'):
                            self.A = layers.Dense(self.n_action, activation=None)(hidden_value)

                        with tf.variable_scope('Q'):
                            out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
                    else:
                        out = layers.Dense(self.n_action, activation=None)(hidden_value)
            return out
        # ---------------- build evaluate_net ------------------
        with tf.name_scope('inputs'):
            self.state_in = tf.placeholder(tf.float32, [None, self.state_dim], name='state_input')
            self.q_target = tf.placeholder(tf.float32, [None, self.n_action], name='Q_target')  # for calculating error
            self.state_ = tf.placeholder(tf.float32, [None, self.state_dim], name='later_state') # later state for computing target #q
            #self.q_eval_wrt_a = tf.placeholder(tf.float32, [None, self.n_action], name='Q_eval')

        #model = keras.Sequential()
        self.q_value = _build_model(self.state_in, self.n_hidden, 'Evaluation_Net')

        # ---------------- build target_net --------------------
        if self.target_net:
            #cnames = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = _build_model(self.state_, self.n_hidden, 'Target_Net')

        # ----- loss and train -----
        with tf.name_scope('loss'):
            if self.prioritized:
                self.ISWeights  = tf.placeholder(tf.float32, [None,1], name='prioritized_weight')
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_value), axis=1)    # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_value))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_value, name='TD_error'))
        with tf.name_scope('trainer'):
            self._train_op = self.trainer.minimize(self.loss)

        # output graph
    def choose_action(self, sess, state):
        # to have batch dimension when feed into tf placeholder
        state = state[np.newaxis, :]

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, self.n_action)  # choose a random action
        else:
            # forward feed the observation and get q value for every actions
            action_value = sess.run(self.q_value, feed_dict={self.state_in: state})
            action = np.argmax(action_value)
        return action

    def learn(self, sess, s, a, r, s_, w=None):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0 and self.target_net:
            sess.run(self.target_replace_op)
            #print('\ntarget_params_replaced\n')

        q_value = sess.run(self.q_value, feed_dict={ self.state_in: s })
        if self.target_net:
            q_next  = sess.run(self.q_next, feed_dict={ self.state_: s_ })
        else:
            q_next  = sess.run(self.q_value, feed_dict={ self.state_in: s_ })

        # change q_target w.r.t q_eval's action

        q_target = q_value.copy()

        batch_index = np.arange(a.shape[0], dtype=np.int32)
        eval_act_index = a.astype(int)

        if self.double_q:
            q_eval4next = sess.run(self.q_value, feed_dict={ self.state_in: s_ })
            max_act4next = np.argmax(q_eval4next,axis=1)
            selected_q_next = q_next[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)

        q_target[batch_index, eval_act_index] = r + self.gamma * selected_q_next

        # train eval network
        if self.prioritized:
            _, abs_errors, loss = sess.run([self._train_op, self.abs_errors, self.loss],
                                        feed_dict={self.state_in: s,
                                                    self.q_target: q_target,
                                                    self.ISWeights: w})
        else:
            _, loss = sess.run([self._train_op, self.loss],
                                        feed_dict={self.state_in: s, self.q_target: q_target})
        # self.loss_his.append(loss)

        # increasing epsilon
        self.eps = self.eps - self.eps_increment if self.eps > self.eps_min else self.eps_min
        self.learn_step_counter += 1

        if self.prioritized:
            return abs_errors, loss
        else:
            return loss

class RL_Table(RL_Agent):   # a q-table learner
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions   # action space is discrete
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.lr    = learning_rate
        self.gamma = reward_decay
        self.eps   = e_greedy   # probability of choosing random action

    def check_state(self, state):  # check the state, if unknown, add to knowledge base
        # in a q-table learner, states need to be known
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, state):
        self.check_state(state)
        # action selection
        if np.random.rand() < self.eps:
            # choose random action
            action = np.random.choice(self.actions)
        else:
            # choose best action
            qsa = self.q_table.loc[state, :]  # columns convert to indices
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(qsa[qsa == np.max(qsa)].index)
        return action

    def learn(self, *args):
        pass

class QLearningTable(RL_Table):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state(s_)   # add next state into the table
        q_predict = self.q_table.loc[s, a]
        if s_ != 'Terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

class SarsaTable(RL_Table):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state(s_)   # add next state into the table
        q_predict = self.q_table.loc[s, a]
        if s_ != 'Terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

class SarsaLambdaTable(RL_Table):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1, trace_decay=0.8):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state(self, state):
        #super(SarsaLambdaTable, self).check_state(self, state)
        if state not in self.q_table.index:
            # append new state to q table
            state_action = pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table.append(state_action)
            # also update eligibility
            self.eligibility_trace.append(state_action)

    def learn(self, s, a, r, s_, a_):
        self.check_state(s_)   # add next state into the table
        q_predict = self.q_table.loc[s, a]
        if s_ != 'Terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        deltaq = (q_target - q_predict)  # update

        # accumulating trace
        self.eligibility_trace.ix[s,a] += 1

        # replacing trace
        self.eligibility_trace.ix[s,:] *= 0
        self.eligibility_trace.ix[s,a]  = 1

        # q update
        self.q_table += self.lr * deltaq * self.eligibility_trace

        # decay trace after update
        self.eligibility_trace *= self.gamma*self.lambda_
