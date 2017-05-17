import keras.backend as K
import tensorflow as tf
from keras import Input
from keras import layers
from keras.engine import Model
from keras.layers import LeakyReLU, Concatenate
from keras.layers.core import Dense
from keras.optimizers import Adam
import numpy as np
from Utils import Action
import Utils as utils

max_turn_angle = 180
min_turn_angle = -180
max_power = 100
min_power = 0


def bound(grad, param, max, min):
    if grad > 0:
        return grad * np.divide(max - param, max - min)
    else:
        return grad * np.divide(param - min, max - min)


def bound_grads(cur_grads, cur_actions, index):
    if index >= 4:
        if index == 4:
            cur_grads[4] = bound(cur_grads[4], cur_actions[index], max_power, min_power)
        elif index == 5:
            cur_grads[5] = bound(cur_grads[5], cur_actions[index], max_turn_angle, min_turn_angle)
        elif index == 6:
            cur_grads[6] = bound(cur_grads[6], cur_actions[index], max_turn_angle, min_turn_angle)
        elif index == 7:
            cur_grads[7] = bound(cur_grads[7], cur_actions[index], max_turn_angle, min_turn_angle)
        elif index == 8:
            cur_grads[8] = bound(cur_grads[8], cur_actions[index], max_power, min_power)
        elif index == 9:
            cur_grads[9] = bound(cur_grads[9], cur_actions[index], max_turn_angle, min_turn_angle)


class CriticNet:
    def __init__(self, sess, tau, learning_rate=0.001, team_size=1, enemy_size=0):
        self.TAU = tau
        self.relu_neg_slope = 0.01
        self.learning_rate = learning_rate
        self.input_size = (58 + (team_size - 1) * 8 + enemy_size * 8) * team_size
        self.sess = sess
        K.set_session(sess)

        self.model, self.action, self.state = self.create_critic_network(self.input_size, 10)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(self.input_size, 10)
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update

    def gradients(self, states, actions):
        grads = self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

        for i in range(len(grads)):
            for j in range(len(grads[i])):
                bound_grads(grads[i], actions[i], j)
        return grads

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        print("Building critic model")
        critic_input_action = Input(shape=[action_dim])
        critic_input_state = Input(shape=[state_size])
        critic_input_final = layers.concatenate([critic_input_state, critic_input_action], axis=1)
        dense1 = Dense(1024, activation='linear', kernel_initializer='he_normal')(critic_input_final)
        relu1 = LeakyReLU(alpha=self.relu_neg_slope)(dense1)
        dense2 = Dense(512, activation='linear', kernel_initializer='he_normal')(relu1)
        relu2 = LeakyReLU(alpha=self.relu_neg_slope)(dense2)
        dense3 = Dense(256, activation='linear', kernel_initializer='he_normal')(relu2)
        relu3 = LeakyReLU(alpha=self.relu_neg_slope)(dense3)
        dense4 = Dense(128, activation='linear', kernel_initializer='he_normal')(relu3)
        relu4 = LeakyReLU(alpha=self.relu_neg_slope)(dense4)
        critic_out = Dense(1, activation='linear', kernel_initializer='he_normal')(relu4)

        model = Model(input=[critic_input_state, critic_input_action], output=critic_out)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model, critic_input_action, critic_input_state
