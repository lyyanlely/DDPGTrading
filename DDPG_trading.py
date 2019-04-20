"""
Created on Fri Mar 20, 2019

@author: Le Yan

to reproduce the model in Xiong2018
"""

#import gym
from __future__ import division
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from RL_superclass import DDPG
from ExpBuffer import Experience_Buffer

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
BUFFERSIZE = 5000
A_BOUND = 100
P_NORM  = price_hist_train.mean()

s_size = 2*NSTOCK+1
a_size = NSTOCK
h_size = 3*NSTOCK

MAX_EPISODE = 200
MAX_EP_STEPS = PERIOD
REC_EPISODE = 20
BATCHSIZE = 32
REC_TRAIN = 20
PRINT_EPI = 20

GAMMA = 0.99
LR_A = 0.001    # learning rate for actor
LR_C = 0.002     # learning rate for critic
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', replace_target_iter=600)
][0]            # you can try different target replacement strategies
e = 1
MODEL_PATH = './model'
OUTPUT_GRAPH = True

# actor and critic
actor_trainer = tf.train.AdamOptimizer(LR_A)
critic_trainer = tf.train.AdamOptimizer(LR_C)

ddpg = DDPG(s_size, a_size, A_BOUND, n_hidden=h_size, reward_decay=GAMMA, replace_method=REPLACEMENT,
            actor_trainer=actor_trainer, critic_trainer=critic_trainer)

# initialization
init = tf.global_variables_initializer()

# replay buffer
Replay_buffer = Experience_Buffer()

# saver
saver = tf.train.Saver()

# game environment
env = pf.stockgame(INIT_VALUE,price_hist_train)
env_test = pf.stockgame(INIT_VALUE,price_hist_test)

#create lists to contain total rewards and steps per episode
jList = []
rList = []
err_list = []
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
        s_in = [np.concatenate(((s[0]-s[0].mean())/(s[0].std()+0.001/NSTOCK),(p-p.mean())/p.std(),np.array([s[1]/s[2]])))]  # normalize price by total
        a = ddpg.choose_action(sess, s_in)
        #v = critic.predict_value(sess, s_in)   #, a/np.mean(np.abs(a),axis=1)
        s1,r,d = env_test.step(a)  #env.step(a[0])
        p1 = env_test.market.current_price

        s1_in = np.concatenate(((s1[0]-s1[0].mean())/(s1[0].std()+0.001/NSTOCK),(p1-p1.mean())/p1.std(),np.array([s1[1]/s1[2]])))
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
            s_in = [np.concatenate(((s[0]-s[0].mean())/(s[0].std()+0.001/NSTOCK),(p-p.mean())/p.std(),np.array([s[1]/s[2]])))]  # normalize price by total
            # a, rnn_state_actor  = actor.choose_action(sess, s_in, rnn_state_actor) #
            # v, rnn_state_critic = critic.predict_value(sess, s_in, a, rnn_state_critic)  #
            a = ddpg.choose_action(sess, s_in)
            # v = critic.predict_value(sess, s_in, a/np.mean(np.abs(a),axis=1))

            a = np.random.normal(a, e)
            # Get new state and reward from environment
            s1,r,d = env.step(a)  #
            p1 = env.market.current_price

            s1_in = np.concatenate(((s1[0]-s1[0].mean())/(s1[0].std()+0.001/NSTOCK),(p1-p1.mean())/p1.std(),np.array([s1[1]/s1[2]])))
            # Add sample to the replay buffer
            Replay_buffer.add([[s_in[0],a,r/s[2]/SIG,s1_in]])
            running_reward = running_reward*GAMMA+r/s[2]/SIG
            s = s1
            p = p1

            if Replay_buffer.size > BUFFERSIZE:
                e *= .9995    # decay the action randomness
                # Sample a random minibatch of N transitions (si , ai , ri , si+1) from R
                replay = Replay_buffer.sample(BATCHSIZE)
                S = np.vstack(replay[:,0])
                A = np.vstack(replay[:,1])
                R = np.vstack(replay[:,2])
                S1= np.vstack(replay[:,3])

                loss = ddpg.learn(sess, S, A, R, S1)

                if j%REC_TRAIN==0:
                    err_list.append(loss)

            # episode stops when
            if d == True or j>=MAX_EP_STEPS:
                if i%PRINT_EPI==0:
                    print("episode:", i, "  reward:", running_reward)
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
                s_in = [np.concatenate(((s[0]-s[0].mean())/(s[0].std()+0.001/NSTOCK),(p-p.mean())/p.std(),np.array([s[1]/s[2]])))]  # normalize price by total
                a = ddpg.choose_action(sess, s_in)
                # v = critic.predict_value(sess, s_in)   #  a/np.mean(np.abs(a),axis=1)
                s1,r,d = env_test.step(a)  #env.step(a[0])
                p1 = env_test.market.current_price

                s1_in = np.concatenate(((s1[0]-s1[0].mean())/(s1[0].std()+0.001/NSTOCK),(p1-p1.mean())/p1.std(),np.array([s1[1]/s1[2]])))
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
        s_in = [np.concatenate(((s[0]-s[0].mean())/(s[0].std()+0.001/NSTOCK),(p-p.mean())/p.std(),np.array([s[1]/s[2]])))]  # normalize price by total
        a = ddpg.choose_action(sess, s_in)
        # v = critic.predict_value(sess, s_in)  #, a/np.mean(np.abs(a),axis=1)
        s1,r,d = env_test.step(a)  #env.step(a[0])
        p1 = env_test.market.current_price

        s1_in = np.concatenate(((s1[0]-s1[0].mean())/(s1[0].std()+0.001/NSTOCK),(p1-p1.mean())/p1.std(),np.array([s1[1]/s1[2]])))
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
