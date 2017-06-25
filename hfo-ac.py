#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

import json

import tensorflow as tf
from keras import backend as K

from ActorNet import ActorNet
from CriticNet import CriticNet
from GameInfo import GameInfo
from Utils import *

np.random.seed(2313)

batch_size = 32  # batch size for training
y = .99  # Discount factor on the target Q-values
startE = 1  # Starting chance of random action
endE = 0.1  # Final chance of random action
evaluate_e = 0  # Epsilon used in evaluation
discount_factor = 0.99
annealing_steps = 10000.  # How many steps of training to reduce startE to endE.
num_episodes = 10000  # How many episodes of game environment to train network with.
pre_train_steps = 1000  # How many steps of random actions before training begins.
num_players = 1
num_opponents = 0
tau = 0.001  # Tau value used in target network update
num_features = (58 + (num_players - 1) * 8 + num_opponents * 8) * num_players
step_counter = 0
load_model = True  # Load the model
use_gpu = False
train = True
if train:
    e = startE
else:
    e = evaluate_e
step_drop = (startE - endE) / annealing_steps

if use_gpu:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
else:
    config = tf.ConfigProto()

exp_buffer = ExpBuffer()
total_reward = 0
sess = tf.Session(config=config)

K.set_session(sess)
actor = ActorNet(team_size=num_players, enemy_size=num_opponents, tau=tau, sess=sess)
critic = CriticNet(team_size=num_players, enemy_size=num_opponents, tau=tau, sess=sess)

# init model by creating new model or loading
# activate HFO agent(s)
# players = []
print("Loading the weights")
if load_model:
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

hfo = HFOEnvironment()
hfo.connectToServer(LOW_LEVEL_FEATURE_SET,
                    '/Users/eclipse/HFO/bin/teams/base/config/formations-dt', 6000,
                    'localhost', 'base_left', False)

for episode in range(num_episodes):
    game_info = GameInfo(1)
    hfo.act(DASH, 0, 0);
    game_info.update(hfo);
    while game_info.status == IN_GAME:
        loss = 0
        # Grab the state features from the environment
        state0 = hfo.getState()
        action_arr = actor.model.predict(np.reshape(state0, [1, num_features]))[0]
        dice = np.random.uniform(0, 1)
        if dice < e and train:
            print "Random action is taken for exploration, e = " + str(e)
            new_action_arr = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1),
                              np.random.uniform(-1, 1), np.random.uniform(-100, 100), np.random.uniform(-180, 180),
                              np.random.uniform(-180, 180), np.random.uniform(-180, 180), np.random.uniform(0, 100),
                              np.random.uniform(-180, 180)]
            action_arr = new_action_arr
        if train and e >= endE and exp_buffer.cur_size >= pre_train_steps:
            e -= step_drop

        # Take an action and get the current game status
        take_action(hfo, get_action(action_arr))
        print action_arr
        game_info.update(hfo)
        state1 = hfo.getState()
        reward = game_info.get_reward()

        # Fill buffer with record
        exp_buffer.add(
            Experience(state0=state0, action=action_arr, state1=state1, reward=reward, done=game_info.episode_over))
        # Train the network
        if exp_buffer.cur_size >= pre_train_steps and train:
            # sample batch
            cur_experience_batch = exp_buffer.sample(batch_size)
            state0s = np.asarray([cur_exp.state0 for cur_exp in cur_experience_batch])
            actions = np.reshape(np.asarray([cur_exp.action for cur_exp in cur_experience_batch]), [batch_size, 10])
            rewards = np.asarray([cur_exp.reward for cur_exp in cur_experience_batch])
            dones = np.asarray([cur_exp.done for cur_exp in cur_experience_batch])
            state1s = np.asarray([cur_exp.state1 for cur_exp in cur_experience_batch])
            y_t = np.zeros((state0s.shape[0], 1))

            target_q_values = critic.target_model.predict([state1s, actor.target_model.predict(state1s)])

            for k in xrange(batch_size):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + discount_factor * target_q_values[k]
            if train:
                loss += critic.model.train_on_batch([state0s, actions], y_t)
                a_for_grad = actor.model.predict(state0s)
                grads = critic.gradients(state0s, a_for_grad)
                actor.update(state0s, grads)
                actor.target_train()
                critic.target_train()

        step_counter += 1

        print("Episode", episode + 1, "Step", step_counter, "Reward", reward, "Loss", loss)

    # Check the outcome of the episode
    total_reward += game_info.total_reward
    print('Episode %d ended with %s' % (episode + 1, hfo.statusToString(game_info.status)))
    print("Episodic TOTAL REWARD @ " + str(episode + 1) + "-th Episode  : " + str(game_info.total_reward))
    print("Total REWARD: ", total_reward, "EOT Reward", game_info.extrinsic_reward)
    if np.mod(episode, 10) == 0:
        actor.model.save_weights("actormodel.h5", overwrite=True)
        with open("actormodel.json", "w") as outfile:
            json.dump(actor.model.to_json(), outfile)

        critic.model.save_weights("criticmodel.h5", overwrite=True)
        with open("criticmodel.json", "w") as outfile:
            json.dump(critic.model.to_json(), outfile)

    # Quit if the server goes down

    if game_info.status == SERVER_DOWN:
        hfo.act(QUIT)
        break
    episode += 1
