import tensorflow as tf
import numpy as np
import os
import sys
import math
from tqdm import tqdm
from simulator import DataSimulator

class SelfAttn(tf.keras.layers.Layer):

    def __init__(self):
        super(SelfAttn, self).__init__()

    def build(self, input_shape):
        # doesn't work rn, gives weights of size batch_sz, input (while we want input, output)
        self.K = self.add_weight("K", input_shape, trainable=True)
        self.Q = self.add_weight("Q", input_shape, trainable=True)
        self.V = self.add_weight("V", input_shape, trainable=True)

    def call(self, inputs):

        keys, queries, values = inputs, inputs, inputs

        K = keys @ self.K
        Q = queries @ self.Q
        V = values @ self.V

        mask_vals = np.triu(np.ones((Q.shape[-2], K.shape[-2])) * np.NINF, k=1)
        mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, Q.shape[-2], K.shape[-2]]), [tf.shape(input=K)[0], 1, 1])
        atten_mtx = tf.nn.softmax(((Q @ tf.transpose(K, [0, 2, 1]))/math.sqrt(K.shape[-1])) + atten_mask)

        return atten_mtx @ V

class Model(tf.keras.Model):

    def __init__(self):

        super(Model, self).__init__()

        self.num_convolutions = 3
        self.convolutions = []

        self.flatten = tf.keras.layers.Flatten()

        self.num_self_attns = 3
        self.self_attns = []

        self.num_dense = 3
        self.dense = []

        for _ in range(self.num_convolutions):
            self.convolutions.append(tf.keras.layers.Convolution2D(5, (4, 4), (1, 2), padding="SAME"))

        for _ in range(self.num_self_attns):
            self.self_attns.append(SelfAttn())

        for _ in range(self.num_dense):
            self.dense.append(tf.keras.layers.Dense(10))

    def call(self, inputs, training=False):

        x = tf.one_hot(inputs, 4, axis=1)

        x = tf.expand_dims(x, -1)

        for conv in self.convolutions:
            x = conv(x)

        x = self.flatten(x)

        for self_attn in self.self_attns:
            x = self_attn(x)

        for dense in self.dense:
            x = dense(x)
        
        return x

if __name__ == '__main__':

    model = Model()
    sim_data = DataSimulator()

    sim_data.add_interactions([('AAAAAAAA', 'CCCCCCCC'), ('CCCCCCCC', 'TTTTTTTT')])

    model.call(sim_data.simulate(100, 20)[0][:5])
        