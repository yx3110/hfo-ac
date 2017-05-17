import keras.backend as K
import tensorflow as tf
from keras import Input
from keras import layers
from keras.engine import Model
from keras.layers import LeakyReLU, Concatenate
from keras.layers.core import Dense
from keras.optimizers import Adam


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
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

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
        dense1 = Dense(1024, activation='linear')(critic_input_final)
        relu1 = LeakyReLU(alpha=self.relu_neg_slope)(dense1)
        dense2 = Dense(512, activation='linear')(relu1)
        relu2 = LeakyReLU(alpha=self.relu_neg_slope)(dense2)
        dense3 = Dense(256, activation='linear')(relu2)
        relu3 = LeakyReLU(alpha=self.relu_neg_slope)(dense3)
        dense4 = Dense(128, activation='linear')(relu3)
        relu4 = LeakyReLU(alpha=self.relu_neg_slope)(dense4)
        critic_out = Dense(1, activation='linear')(relu4)

        model = Model(input=[critic_input_state, critic_input_action], output=critic_out)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model, critic_input_action, critic_input_state