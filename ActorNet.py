import Utils as utils
import keras as keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
import numpy as np


class ActorNet:
    def __init__(self, learning_rate=0.001, team_size=1, enemy_size=0):
        relu_neg_slope = 0.01
        self.input_size = (58 + (team_size - 1) * 8 + enemy_size * 8) * team_size
        self.actor_input = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
        self.actor_target = tf.placeholder(dtype=tf.float32, name="actor_target")

        self.dense1 = Dense(1024, activation=keras.layers.advanced_activations.LeakyReLU(alpha=relu_neg_slope)) \
            (self.actor_input)
        self.dense2 = Dense(512, activation=keras.layers.advanced_activations.LeakyReLU(alpha=relu_neg_slope)) \
            (self.dense1)
        self.dense3 = Dense(256, activation=keras.layers.advanced_activations.LeakyReLU(alpha=relu_neg_slope)) \
            (self.dense2)
        self.dense4 = Dense(128, activation=keras.layers.advanced_activations.LeakyReLU(alpha=relu_neg_slope)) \
            (self.dense3)
        self.actor_out = Dense(10, activation=keras.layers.advanced_activations.LeakyReLU(alpha=relu_neg_slope)) \
            (self.dense4)

        self.actor_loss = tf.squared_difference(self.actor_out, self.actor_target)

        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op_actor = self.actor_optimizer.apply_gradients(
            self.actor_target_grads, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess):
        action_arr = sess.run(self.actor_out, {self.actor_input: np.reshape(state, [1, self.input_size])})
        return action_arr

    def update(self, state, target, action, sess=None):
        feed_dict = {self.actor_input: np.reshape(state, [1, self.input_size]), self.actor_target: target,
                     self.actor_out: action}
        _, loss = sess.run([self.train_op_actor, self.actor_loss], feed_dict)
        return loss
