import tensorflow as tf
import numpy as np
from keras.layers.core import Dense, Dropout, Activation
import keras as keras


class CriticNet:
    def __init__(self, learning_rate=0.001, team_size=1, enemy_size=0):
        relu_neg_slope = 0.01
        self.input_size = (58 + (team_size - 1) * 8 + enemy_size * 8) * team_size + 10 * team_size
        self.critic_input = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
        self.critic_target = tf.placeholder(dtype=tf.float32, name="critic_target")

        self.dense1 = Dense(1024, activation=keras.layers.advanced_activations.LeakyReLU(alpha=relu_neg_slope)) \
            (self.critic_input)
        self.dense2 = Dense(512, activation=keras.layers.advanced_activations.LeakyReLU(alpha=relu_neg_slope)) \
            (self.dense1)
        self.dense3 = Dense(256, activation=keras.layers.advanced_activations.LeakyReLU(alpha=relu_neg_slope)) \
            (self.dense2)
        self.dense4 = Dense(128, activation=keras.layers.advanced_activations.LeakyReLU(alpha=relu_neg_slope)) \
            (self.dense3)
        self.critic_out = Dense(1, activation=keras.layers.advanced_activations.LeakyReLU(alpha=relu_neg_slope)) \
            (self.dense4)
        self.critic_loss = tf.squared_difference(self.critic_out, self.critic_target)

        self.grads_wrt_input_tensor = tf.gradients(self.critic_loss, self.critic_input)[0]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op_critic = self.optimizer.minimize(
            self.critic_loss)

    def predict(self, state, action, sess=None):
        cur_input = np.reshape(np.append(state, action), [1, self.input_size])

        return sess.run(self.critic_out, {self.critic_input: cur_input})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        state = np.reshape(state, [1, self.input_size])
        feed_dict = {self.critic_input: state, self.critic_target: target}
        _, loss = sess.run([self.train_op_critic, self.critic_loss], feed_dict)
        return loss

    def grads_wrt_input(self, state, target, sess=None):
        state = np.reshape(state, [1, self.input_size])
        feed_dict = {self.critic_input: state, self.critic_target: target}
        _, gradient = sess.run([self.train_op_critic, self.grads_wrt_input_tensor], feed_dict)
        return gradient
