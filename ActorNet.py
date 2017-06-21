import keras.backend as K
import tensorflow as tf
from keras.layers import Input
from keras.engine import Model
from keras.layers import LeakyReLU
from keras.layers.core import Dense
from keras import layers
from keras import initializers

max_turn_angle = 180
min_turn_angle = -180
max_power = 100
min_power = 0


class ActorNet:
    def __init__(self, sess, tau, learning_rate=0.00001, team_size=1, enemy_size=0, ):
        self.sess = sess
        self.TAU = tau
        K.set_session(sess)

        self.input_size = (58 + (team_size - 1) * 8 + enemy_size * 8) * team_size
        self.relu_neg_slope = 0.01
        self.model, self.weights, self.state = self.create_actor_network(self.input_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(self.input_size)

        self.action_gradient = tf.placeholder(tf.float32, [None, 10])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def update(self, state, grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: state,
            self.action_gradient: grads
        })

    def create_actor_network(self, state_size):
        print("Building actor model")
        actor_input = Input(shape=[state_size])
        dense1 = Dense(1024, activation='linear', kernel_initializer=initializers.he_normal(seed=0.01))(actor_input)
        relu1 = LeakyReLU(alpha=self.relu_neg_slope)(dense1)
        dense2 = Dense(512, activation='linear', kernel_initializer=initializers.he_normal(seed=0.01))(relu1)
        relu2 = LeakyReLU(alpha=self.relu_neg_slope)(dense2)
        dense3 = Dense(256, activation='linear', kernel_initializer=initializers.he_normal(seed=0.01))(relu2)
        relu3 = LeakyReLU(alpha=self.relu_neg_slope)(dense3)
        dense4 = Dense(128, activation='linear', kernel_initializer=initializers.he_normal(seed=0.01))(relu3)
        relu4 = LeakyReLU(alpha=self.relu_neg_slope)(dense4)
        action_out = Dense(4, activation='softmax', kernel_initializer=initializers.he_normal(seed=0.01))(relu4)
        param_out = Dense(6, activation='linear', kernel_initializer=initializers.he_normal(seed=0.01))(relu4)
        actor_out = layers.concatenate([action_out, param_out], axis=1)
        model = Model(inputs=actor_input, outputs=actor_out)
        return model, model.trainable_weights, actor_input
