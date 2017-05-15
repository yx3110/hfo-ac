#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

import itertools
from EXPBuffer import ExpBuffer
from hfo import *

import Utils as utils
import tensorflow as tf
from CriticNet import CriticNet
from ActorNet import ActorNet
import random

path = "./hfo-ac/"  # The path to save our model to.
if not os.path.exists(path):
    os.makedirs(path)

batch_size = 32  # batch size for training
y = .99  # Discount factor on the target Q-values
startE = 1  # Starting chance of random action
endE = 0.1  # Final chance of random action
discount_factor = 0.99
annealing_steps = 10000.  # How many steps of training to reduce startE to endE.
num_episodes = 10000  # How many episodes of game environment to train network with.
pre_train_steps = 100  # How many steps of random actions before training begins.
num_players = 1
num_opponents = 0
tau = 0.001  # Tau value used in target network update
num_features = (58 + (num_players - 1) * 8 + num_opponents * 8) * num_players
step_counter = 0
load_model = True  # Load the model
num_games = 0
e = startE
stepDrop = (startE - endE) / annealing_steps

tf.reset_default_graph()
exp_buffer = ExpBuffer()
with tf.Session() as sess:
    actor = ActorNet(team_size=num_players, enemy_size=num_opponents)
    critic = CriticNet(team_size=num_players, enemy_size=num_opponents)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # init model by creating new model or loading
    if load_model:
        saver.restore(sess, path)
        print 'session restored'
    else:
        sess.run(init)
    # activate HFO agent(s)
    # players = []
    for i in range(num_players):
        hfo = HFOEnvironment()
        hfo.connectToServer(LOW_LEVEL_FEATURE_SET,
                            '/Users/eclipse/HFO/bin/teams/base/config/formations-dt', 6000,
                            'localhost', 'base_left', False)
        # players.append(hfo)
    from lib import plotting

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for episode in itertools.count():
        game_status = IN_GAME
        num_games += 1
        while game_status == IN_GAME:
            # Grab the state features from the environment
            state = hfo.getState()
            action_arr = actor.predict(state, sess)
            candidate_action = utils.get_action(action_arr)
            dice = random.uniform(0, 1)
            if dice < e:
                new_candidate = random.randint(0, 3)
                while new_candidate == 2:
                    new_candidate = random.randint(0, 3)
                candidate_action.action = new_candidate
            if e >= endE:
                e -= stepDrop
            # Take an action and get the current game status
            utils.take_action(hfo, candidate_action)
            game_status = hfo.step()
            newState = hfo.getState()
            reward = utils.calculate_reward(state, newState, game_status)
            value = critic.predict(state, action_arr, sess)
            stats.episode_rewards[episode] += reward
            stats.episode_lengths[episode] = num_games
            '''
            Fill buffer with record
            '''
            exp_buffer.add(utils.Experience(prev_state=state, action=action_arr, cur_state=newState, reward=reward))
            print("exp size: " + str(exp_buffer.cur_size))
            if exp_buffer.cur_size >= pre_train_steps:
                '''
                Train the network
                '''
                # sample batch
                cur_experience_batch = exp_buffer.sample(batch_size)

                for i in range(batch_size):
                    cur_exp = cur_experience_batch[i]
                    value_next = critic.predict(cur_exp.prev_state, cur_exp.action, sess)
                    td_target = cur_exp.reward + discount_factor * value_next
                    td_error = td_target - critic.predict(cur_exp.cur_state,
                                                          actor.predict(cur_exp.cur_state, sess=sess),
                                                          sess)
                    critic.update(np.append(cur_exp.prev_state, cur_exp.action), td_target, sess)

                    actor.update(cur_exp.prev_state, td_error, cur_exp.action, sess)
            step_counter += 1
            if step_counter % 60 == 0:
                save_path = saver.save(sess, path)
                print("Model saved in file: %s" % save_path)
        # Check the outcome of the episode

        print('Episode %d ended with %s' % (episode, hfo.statusToString(game_status)))
        # Quit if the server goes down

        if game_status == SERVER_DOWN:
            hfo.act(QUIT)
            break
