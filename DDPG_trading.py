"""
Created on Fri Mar 20, 2019

@author: Le Yan

to reproduce the model in Xiong2018

state input: holding state, portfolio, price, cash_value
action output: holding state, portfolio, sum(portfolio)=1
reward: cash > expected stock
"""

#import gym
from __future__ import division
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from RL_superclass import DDPG
from ExpBuffer import Experience_Buffer, Prioritized_Buffer

import threading
import multiprocessing
import scipy.signal
#%matplotlib inline

from random import choice
from time import sleep
from time import time

from tensorflow.keras import backend as K

import portfolio as pf
import os
from fnmatch import filter
from read_stock_json import price_history

plt.ion()

reload(pf)

# import data
dirc  = 'stock/'

price_hist_train, price_hist_test, names, dates = price_history(dirc,nstock=100)

# global parameters
NSTOCK = price_hist_train.shape[0]
T = price_hist_train.shape[1]

PERIOD = 500
SIG = .03
INIT_VALUE = 10000
BUFFERSIZE = 20000
#A_BOUND = 100
P_NORM  = price_hist_train.mean()

s_size = 2*NSTOCK+1
a_size = NSTOCK   # holding value fraction, distribution in stocks (NSTOCK bandit?)
h_size_a = [500, 400]
h_size_c = [700, 20]

MAX_EPISODE = 2000
MAX_EP_STEPS = PERIOD
REC_EPISODE = 200
BATCHSIZE = 64
REC_TRAIN = 200
PRINT_EPI = 5

GAMMA = 0.99
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0002     # learning rate for critic
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies
e = 1
e_min = 0.01
MODEL_PATH = './model'
OUTPUT_GRAPH = True

# actor and critic
actor_trainer = tf.train.RMSPropOptimizer(LR_A)
critic_trainer = tf.train.AdamOptimizer(LR_C)

ddpg = DDPG(s_size, a_size, n_hidden_a=h_size_a, n_hidden_c=h_size_c, reward_decay=GAMMA, replace_method=REPLACEMENT,
            prioritized=True,
            actor_trainer=actor_trainer, critic_trainer=critic_trainer)

# initialization
init = tf.global_variables_initializer()

# replay buffer
Replay_buffer = Prioritized_Buffer(buffer_size=BUFFERSIZE)

# saver
saver = tf.train.Saver()

# game environment
env = pf.stockgame(INIT_VALUE,price_hist_train)
env_test = pf.stockgame(INIT_VALUE,price_hist_test)

#create lists to contain total rewards and steps per episode
jList = []
rList = []
err_list = []
loss = 10000
prof_list_ep = []
with tf.Session() as sess:

    if OUTPUT_GRAPH:  # output the graph
        tf.summary.FileWriter("logs/", sess.graph)

    sess.run(init)
    tflag = 0

    # random initial strategy
    prof_list_0 = []
    running_reward = 0
    time_len = price_hist_test.shape[1]
    env_test.initialize_game(INIT_VALUE, time_len)
    s = env_test.portfolio.current_state()
    p = env_test.market.current_price
    prof_list_0.append(s[2])
    for t in xrange(time_len):
        s_in = np.concatenate((s[0]/(s[0].max()+0.0001),p/p.max(),np.array([s[1]/s[2]])))  # normalize price by total
        a = ddpg.choose_action(sess, s_in)
        #chold = 1-s[1]/s[2]  # current hold
        # a = np.clip(np.random.normal(a, a*e),0,1)
        action = (s[2]*a[0]*a[1:]/p-s[0]).astype(np.int)  # change in hold
        #v = critic.predict_value(sess, s_in)   #, a/np.mean(np.abs(a),axis=1)
        s1,r,d = env_test.step(action)  #env.step(a[0])
        p1 = env_test.market.current_price

        s1_in = np.concatenate((s1[0]/(s1[0].max()+0.0001),p1/p1.max(),np.array([s1[1]/s1[2]])))
            # Add sample to the replay buffer
        running_reward = running_reward*GAMMA+r/s[2]/SIG
        s = s1
        p = p1
        prof_list_0.append(s[2])

    for i in range(MAX_EPISODE):
        #Reset environment and get first new observation
        running_reward = 0   # record total reward

        # if i>10/(LR_A*MAX_EP_STEPS):
        #     tflag = 1

        env.initialize_game(INIT_VALUE, PERIOD)
        s = env.portfolio.current_state()
        p = env.market.current_price
        # rnn_state_actor = actor.state_init
        # rnn_state_critic= critic.state_init
        # batch_rnn_actor = rnn_state_actor
        # batch_rnn_critic= rnn_state_critic

        d = False  # flag for ending of the investment

        j = 0
        #The Q-Network
        while j < MAX_EP_STEPS:
            j+=1
            # Choose an action by the actor network:
            s_in = np.concatenate((s[0]/(s[0].max()+0.0001),p/p.max(),np.array([s[1]/s[2]])))
            # [np.concatenate(((s[0]-s[0].mean())/(s[0].std()+0.001/NSTOCK),(p-p.mean())/p.std(),np.array([s[1]/s[2]])))]  # normalize price by total
            # a, rnn_state_actor  = actor.choose_action(sess, s_in, rnn_state_actor) #
            # v, rnn_state_critic = critic.predict_value(sess, s_in, a, rnn_state_critic)  #
            a = ddpg.choose_action(sess, s_in)
            # v = critic.predict_value(sess, s_in, a/np.mean(np.abs(a),axis=1))

            a = np.clip(np.random.normal(a, a*e),0,1)
            action = (s[2]*a[0]*a[1:]/p-s[0]).astype(np.int)
            # Get new state and reward from environment
            s1,r,d = env.step(action)  #
            p1 = env.market.current_price

            s1_in = np.concatenate((s1[0]/(s1[0].max()+0.0001),p1/p1.max(),np.array([s1[1]/s1[2]])))
            # np.concatenate(((s1[0]-s1[0].mean())/(s1[0].std()+0.001/NSTOCK),(p1-p1.mean())/p1.std(),np.array([s1[1]/s1[2]])))
            # Add sample to the replay buffer
            Replay_buffer.add(np.array([[s_in,a,r/s[2]/SIG,s1_in]]))
            running_reward = running_reward*GAMMA+r/s[2]/SIG
            s = s1
            p = p1

            if Replay_buffer.size >= BUFFERSIZE:
                e = max(e*.99999, e_min)    # decay the action randomness
                # Sample a random minibatch of N transitions (si , ai , ri , si+1) from R
                tree_idx, replay, ISWeights = Replay_buffer.sample(BATCHSIZE)
                S = np.vstack(replay[:,0])
                A = np.vstack(replay[:,1])
                R = np.vstack(replay[:,2])
                S1= np.vstack(replay[:,3])

                loss, abs_td = ddpg.learn(sess, S, A, R, S1, ISWeights)
                Replay_buffer.batch_update(tree_idx, abs_td)

                if j%REC_TRAIN==0:
                    err_list.append(loss)

            # episode stops when
            if d == True or j>=MAX_EP_STEPS:
                if i%PRINT_EPI==0:
                    print("episode:", i, "  reward:", running_reward, "   exploration:", e, "   loss:", loss)
                break

        # if j==num_steps:
        # #Reduce chance of random action as we train the model.
        # 	e = 1./((i/50) + 10)
        if  i%REC_EPISODE==0:
            prof_list = []
            running_reward = 0
            time_len = price_hist_test.shape[1]
            env_test.initialize_game(INIT_VALUE, time_len)
            s = env_test.portfolio.current_state()
            p = env_test.market.current_price
            prof_list.append(s[2])
            for t in xrange(time_len):
                s_in = np.concatenate((s[0]/(s[0].max()+0.0001),p/p.max(),np.array([s[1]/s[2]])))  # normalize price by total
                a = ddpg.choose_action(sess, s_in)
                # v = critic.predict_value(sess, s_in)   #  a/np.mean(np.abs(a),axis=1)

                action = (s[2]*a[0]*a[1:]/p-s[0]).astype(np.int)
                s1,r,d = env_test.step(action)  #env.step(a[0])
                p1 = env_test.market.current_price

                s1_in = np.concatenate((s1[0]/(s1[0].max()+0.0001),p1/p1.max(),np.array([s1[1]/s1[2]])))
                # Add sample to the replay buffer
                running_reward = running_reward*GAMMA+r/s[2]/SIG
                s = s1
                p = p1
                prof_list.append(s[2])
            prof_list_ep.append(prof_list)

            # save model
            saver.save(sess,MODEL_PATH+'/model-Episode'+str(i)+'.cptk')
            print ("Saved Model")

        # Record episodes (training course)
        jList.append(j)
        rList.append(running_reward)

    prof_list = []
    running_reward = 0
    time_len = price_hist_test.shape[1]
    env_test.initialize_game(INIT_VALUE, time_len)
    s = env_test.portfolio.current_state()
    p = env_test.market.current_price
    prof_list.append(s[2])
    for t in xrange(time_len):
        s_in = np.concatenate((s[0]/(s[0].max()+0.0001),p/p.max(),np.array([s[1]/s[2]])))  # normalize price by total
        a = ddpg.choose_action(sess, s_in)
        # v = critic.predict_value(sess, s_in)  #, a/np.mean(np.abs(a),axis=1)

        action = (s[2]*a[0]*a[1:]/p-s[0]).astype(np.int)
        s1,r,d = env_test.step(action)  #env.step(a[0])
        p1 = env_test.market.current_price

        s1_in = np.concatenate((s1[0]/(s1[0].max()+0.0001),p1/p1.max(),np.array([s1[1]/s1[2]])))
            # Add sample to the replay buffer
        running_reward = running_reward*GAMMA+r/s[2]/SIG
        s = s1
        p = p1
        prof_list.append(s[2])

    prof_list_ep.append(prof_list)
    # save model
    saver.save(sess,MODEL_PATH+'/model-Episode'+str(MAX_EPISODE)+'.cptk')
    print ("Saved Model")


print "Average episode return: " + str(sum(rList)/MAX_EPISODE)


plt.ion()
