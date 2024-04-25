import tensorflow as tf
import numpy as np
import os
import sys
import math
from tqdm import tqdm
from simulator import DataSimulator

NUM_EPOCHS = 10
BATCH_SZ = 32

# add more weight to positives in BCE loss

class SelfAttn(tf.keras.layers.Layer):

    def __init__(self, in_shape, out_shape, num_heads):
        super(SelfAttn, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.num_heads = num_heads

    def build(self, input_shape):
        # doesn't work rn, gives weights of size batch_sz, input (while we want input, output)

        self.K = [self.add_weight(f'K{i}', (self.in_shape, self.out_shape), trainable=True) for i in range(self.num_heads)]
        self.Q = [self.add_weight(f'Q{i}', (self.in_shape, self.out_shape), trainable=True) for i in range(self.num_heads)]
        self.V = [self.add_weight(f'V{i}', (self.in_shape, self.out_shape), trainable=True) for i in range(self.num_heads)]

        self.W = self.add_weight(f'W', (self.out_shape*self.num_heads, self.out_shape))

    def call(self, inputs):

        keys, queries, values = inputs, inputs, inputs

        K = [keys @ k for k in self.K]
        Q = [queries @ q for q in self.Q]
        V = [values @ v for v in self.V]

        mask_vals = np.triu(np.ones((Q[0].shape[-2], K[0].shape[-2])) * np.NINF, k=1)
        mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, Q[0].shape[-2], K[0].shape[-2]]), [tf.shape(input=K[0])[0], 1, 1])

        single_head_attns = tf.concat([tf.nn.softmax(((q @ tf.transpose(k, [0, 2, 1]))/math.sqrt(k.shape[-1])) + atten_mask) @ v for q, k, v in zip(Q, K, V)], axis=-1)

        return single_head_attns @ self.W

class Model(tf.keras.Model):

    def __init__(self, seq_len):

        super(Model, self).__init__()

        self.num_convolutions = 1
        self.convolutions = []

        self.seq_len = seq_len

        self.flatten = tf.keras.layers.Flatten()

        self.num_self_attns = 1
        self.self_attns = []

        self.num_dense = 2
        self.dense = []

        for _ in range(self.num_convolutions):
            self.convolutions.append(tf.keras.layers.Convolution2D(5, (4, 20), (1, 1), padding="SAME"))

        self.convolutions.append(tf.keras.layers.Convolution2D(1, (4, 20), (1, 1), padding="SAME"))
        self.convolutions.append(tf.keras.layers.MaxPool2D((1, 20), (1, 5)))
        self.conv_out_shape = (self.seq_len - 20)//5 + 1

        self.self_attns.append(SelfAttn(self.conv_out_shape, 30, 5))
        for _ in range(self.num_self_attns):
            self.self_attns.append(SelfAttn(30, 30, 5))

        for _ in range(self.num_dense):
            self.dense.append(tf.keras.layers.Dense(10, activation='leaky_relu'))
        
        self.dense.append(tf.keras.layers.Dense(2, activation='softmax'))

    def call(self, inputs, training=False):

        x = tf.one_hot(inputs, 4, axis=1)

        # adding "channel" for conv2D,
        x = tf.expand_dims(x, -1)

        for conv in self.convolutions:
            x = conv(x)

        x = tf.reshape(x, (-1, 4, self.conv_out_shape))

        for self_attn in self.self_attns:
            x = self_attn(x)

        x = self.flatten(x)

        for dense in self.dense:
            x = dense(x)
        
        return x

if __name__ == '__main__':

    model = Model(300)
    sim_data = DataSimulator()

    sim_data.add_interactions([('AAAAAAAA', 'CCCCCCCC'), ('CCCCCCCC', 'TTTTTTTT')])

    pos, neg, pos_labels, neg_labels = sim_data.simulate(300, 10000)

    train_X = tf.concat([pos, neg], axis=0)
    train_y = tf.concat([pos_labels, neg_labels], axis=0)

    inds = tf.random.shuffle(tf.range(train_X.shape[0]))

    print(train_y)
    train_y = tf.one_hot(train_y,2)

    model.call(train_X)

    train_X = tf.gather(train_X, inds)
    train_y = tf.gather(train_y, inds)

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.categorical_accuracy]
    )

    model.fit(train_X, train_y, BATCH_SZ, NUM_EPOCHS)

    test_pos, test_neg, test_pos_labels, test_neg_labels = sim_data.simulate(300, BATCH_SZ)

    test_X = tf.concat([test_pos, test_neg], axis=0)
    test_y = tf.concat([test_pos_labels, test_neg_labels], axis=0)

    test_y = tf.one_hot(test_y,2)

    inds = tf.random.shuffle(tf.range(test_X.shape[0]))

    test_X = tf.gather(test_X, inds)
    test_y = tf.gather(test_y, inds)

    print(model.test_on_batch(test_X, test_y))

    print(test_y[0], model.call(tf.expand_dims(test_X[0], 0)))
    print(test_y[5], model.call(tf.expand_dims(test_X[5], 0)))
    print(test_y[7], model.call(tf.expand_dims(test_X[7], 0)))
        