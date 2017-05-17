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
import json

path = "./hfo-ac/"  # The path to save our model to.
if not os.path.exists(path):
    os.makedirs(path)

batch_size = 32  # batch size for training
y = .99  # Discount factor on the target Q-values
startE = 1  # Starting chance of random action
endE = 0.05  # Final chance of random action
discount_factor = 0.99
annealing_steps = 10000.  # How many steps of training to reduce startE to endE.
num_episodes = 10000  # How many episodes of game environment to train network with.
pre_train_steps = 1000  # How many steps of random actions before training begins.
num_players = 1
num_opponents = 0
tau = 0.001  # Tau value used in target network update
num_features = (58 + (num_players - 1) * 8 + num_opponents * 8) * num_players
step_counter = 0
load_model = False  # Load the model
train = True
num_games = 0
e = startE
stepDrop = (startE - endE) / annealing_steps

tf.reset_default_graph()
exp_buffer = ExpBuffer()
total_reward = 0
with tf.Session() as sess:
    actor = ActorNet(team_size=num_players, enemy_size=num_opponents, sess=sess, tau=tau)
    critic = CriticNet(team_size=num_players, enemy_size=num_opponents, tau=tau, sess=sess)

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
    print("Now we load the weight")
    if load_model:
        try:
            actor.model.load_weights("actormodel.h5")
            critic.model.load_weights("criticmodel.h5")
            actor.target_model.load_weights("actormodel.h5")
            critic.target_model.load_weights("criticmodel.h5")
            print("Weight load successfully")
        except:
            print("Cannot find the weight")

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

    for episode in range(10000):
        episode_total_reward = 0
        game_status = IN_GAME
        num_games += 1
        while game_status == IN_GAME:
            loss = 0
            # Grab the state features from the environment
            state = hfo.getState()
            action_arr = actor.model.predict(np.reshape(state, [1, num_features]))[0]
            candidate_action = utils.get_action(action_arr)
            dice = random.uniform(0, 1)
            if dice < e:
                new_candidate_action = random.randint(0, 3)
                while new_candidate_action == 2:
                    new_candidate_action = random.randint(0, 3)
                candidate_action.action = new_candidate_action
                if new_candidate_action == 0 or new_candidate_action == 3:
                    candidate_action.param1 = random.uniform(-100, 100)
                    candidate_action.param2 = random.uniform(-180, 180)
                else:
                    candidate_action.param1 = random.uniform(-180, 180)
                    candidate_action.param2 = 0
            if e >= endE:
                e -= stepDrop
            # Take an action and get the current game status
            utils.take_action(hfo, candidate_action)
            print action_arr
            game_status = hfo.step()
            new_state = hfo.getState()
            reward = utils.calculate_reward(state, new_state, game_status)
            done = game_status != IN_GAME
            stats.episode_rewards[episode] += reward
            stats.episode_lengths[episode] = num_games
            '''
            Fill buffer with record
            '''
            exp_buffer.add(
                utils.Experience(prev_state=state, action=action_arr, cur_state=new_state, reward=reward, done=done))
            print("exp size: " + str(exp_buffer.cur_size))
            # Train the network
            if exp_buffer.cur_size >= pre_train_steps and train:
                # sample batch
                cur_experience_batch = exp_buffer.sample(batch_size)
                prev_states = np.asarray([cur_exp.prev_state for cur_exp in cur_experience_batch])
                actions = np.reshape(np.asarray([cur_exp.action for cur_exp in cur_experience_batch]), [batch_size, 10])
                rewards = np.asarray([cur_exp.reward for cur_exp in cur_experience_batch])
                dones = np.asarray([cur_exp.done for cur_exp in cur_experience_batch])
                cur_states = np.asarray([cur_exp.cur_state for cur_exp in cur_experience_batch])
                y_t = np.asarray([cur_exp.reward for cur_exp in cur_experience_batch])

                target_q_values = critic.target_model.predict([cur_states, actor.target_model.predict(cur_states)])
                for k in range(batch_size):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + discount_factor * target_q_values[k]

                critic.model.train_on_batch([prev_states, actions], y_t)
                a_for_grad = actor.model.predict(prev_states)
                grads = critic.gradients(prev_states, a_for_grad)
                actor.update(prev_states, grads)
                actor.target_train()
                critic.target_train()

            step_counter += 1
            episode_total_reward += reward

            print("Episode", i, "Step", step_counter, "Reward", reward)

        # Check the outcome of the episode
        total_reward += episode_total_reward
        print('Episode %d ended with %s' % (episode, hfo.statusToString(game_status)))
        print("Episodic TOTAL REWARD @ " + str(episode) + "-th Episode  : " + str(episode_total_reward))
        print("Total REWARD: " + str(total_reward))
        if np.mod(episode, 5) == 0:
            save_path = saver.save(sess, path)
            print("Model saved in file: %s" % save_path)
            actor.model.save_weights("actormodel.h5", overwrite=True)
            with open("actormodel.json", "w") as outfile:
                json.dump(actor.model.to_json(), outfile)

            critic.model.save_weights("criticmodel.h5", overwrite=True)
            with open("criticmodel.json", "w") as outfile:
                json.dump(critic.model.to_json(), outfile)

        # Quit if the server goes down

        if game_status == SERVER_DOWN:
            hfo.act(QUIT)
            break
