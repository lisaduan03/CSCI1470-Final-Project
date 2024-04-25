import tensorflow as tf
import numpy as np
import os
import pandas as pd
import sys
import math
from tqdm import tqdm
from simulator import DataSimulator, map_encode
import random


NUM_EPOCHS = 10
BATCH_SZ = 32

# add more weight to positives in BCE loss

class SelfAttn(tf.keras.layers.Layer):
    '''
    Standard multi-headed self attention layer
    '''

    def __init__(self, in_shape, out_shape, num_heads):
        '''
        Initialize self attention layer.

        Parameters:
        -----------
        in_shape: input window size
        out_shape: output window size
        num_heads: number of self attention heads

        Returns:
        --------
        None
        '''
        super(SelfAttn, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.num_heads = num_heads

    def build(self, input_shape=None):
        '''
        Build weights for the self attention layer.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        '''
        # make list of weight matrices, one for each head (K, Q, V)
        self.K = [self.add_weight(f'K{i}', (self.in_shape, self.out_shape), trainable=True) for i in range(self.num_heads)]
        self.Q = [self.add_weight(f'Q{i}', (self.in_shape, self.out_shape), trainable=True) for i in range(self.num_heads)]
        self.V = [self.add_weight(f'V{i}', (self.in_shape, self.out_shape), trainable=True) for i in range(self.num_heads)]

        # weight matrix for combining head information
        self.W = self.add_weight(f'W', (self.out_shape*self.num_heads, self.out_shape))

    def call(self, inputs):
        '''
        Self attention layer call, with masking as per HW5.

        Parameters:
        -----------
        inputs - inputs over which to compute self attention

        Returns:
        --------
        Result of self attention computation
        '''

        keys, queries, values = inputs, inputs, inputs

        # computing keys, queries, values
        K = [keys @ k for k in self.K]
        Q = [queries @ q for q in self.Q]
        V = [values @ v for v in self.V]

        # computing mask
        mask_vals = np.triu(np.ones((Q[0].shape[-2], K[0].shape[-2])) * np.NINF, k=1)
        mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, Q[0].shape[-2], K[0].shape[-2]]), [tf.shape(input=K[0])[0], 1, 1])

        # computing a list of single head attentions, concatenated as a tensor
        single_head_attns = tf.concat([tf.nn.softmax(((q @ tf.transpose(k, [0, 2, 1]))/math.sqrt(k.shape[-1])) + atten_mask) @ v for q, k, v in zip(Q, K, V)], axis=-1)

        # combine single head information via dense layer
        return single_head_attns @ self.W

class Model(tf.keras.Model):
    '''
    Our take on the SATORI model!
    '''

    def __init__(self, seq_len):
        '''
        Set up all necessary layers for model call step.

        Parameters:
        -----------
        seq_len - the length of the input sequences (necessary to deal with 
        None batch size during training)

        Returns:
        --------
        None
        '''

        super(Model, self).__init__()

        self.num_convolutions = 1
        self.convolutions = []

        self.seq_len = seq_len

        self.flatten = tf.keras.layers.Flatten()

        self.num_self_attns = 1
        self.self_attns = []

        self.num_dense = 2
        self.dense = []

        # if we change padding to VALID, will have to compute conv_out_shape 
        # successively for each layer so that we get a correct out_shape to 
        # pass into self attention layer weight build
        for _ in range(self.num_convolutions):
            # note that the 4 here is necessary to make this convolution 
            # essentially 1D, but the 20 and 5 are arbitrary hyperparameters 
            # (although the 20 param should match what we expect to be our 
            # motif lengths, at least for first conv layer)
            self.convolutions.append(tf.keras.layers.Convolution2D(5, (4, 20), (1, 1), padding="SAME"))

        # arbitrary second conv
        self.convolutions.append(tf.keras.layers.Convolution2D(1, (4, 20), (1, 1), padding="SAME"))
        self.convolutions.append(tf.keras.layers.MaxPool2D((1, 20), (1, 5)))

        # 20 and 5 are coming from the max pooling layer here, given prior two
        # convolutions both keep same dimensions
        self.conv_out_shape = (self.seq_len - 20)//5 + 1

        # 30 is an arbitrary hyperparam here, feel free to change
        # 5 is also arbitrary (num of heads)
        self.self_attns.append(SelfAttn(self.conv_out_shape, 30, 5))
        for _ in range(self.num_self_attns):
            self.self_attns.append(SelfAttn(30, 30, 5))

        # more arbitrary hyperparams
        for _ in range(self.num_dense):
            self.dense.append(tf.keras.layers.Dense(10, activation='leaky_relu'))
        
        # can do 2, softmax (works easier with loss function I've put in) or
        # 1, sigmoid...both should theoretically work, but may have to code 
        # custom loss func as keras BCEs are a little finnicky
        self.dense.append(tf.keras.layers.Dense(2, activation='softmax'))

    def call(self, inputs, training=False):
        '''
        Call function for our SATORI model. Does 1D convolutions, gets rid of 
        the channel dimension, then does self attention and dense layers. 

        Parameters:
        -----------
        inputs - inputs over which to predict label distribution
        training - flag we might use if we do regularization

        Returns:
        --------
        Resulting enhancer label distribution
        '''

        x = tf.one_hot(inputs, 4, axis=1)

        # adding "channel" for conv2D,
        x = tf.expand_dims(x, -1)

        for conv in self.convolutions:
            x = conv(x)

        # removing "channel" for conv2D
        x = tf.reshape(x, (-1, 4, self.conv_out_shape))

        for self_attn in self.self_attns:
            x = self_attn(x)

        x = self.flatten(x)

        for dense in self.dense:
            x = dense(x)
        
        return x

if __name__ == '__main__':

    # sequence length 300 is arbitrary here
    model = Model(300)

    if sys.argv[1] == 'simulate':
        sim_data = DataSimulator()

        # motifs also arbitrary, maybe try pushing up to match convolution length 
        # dim of 20
        sim_data.add_interactions([('AAAAAAAA', 'CCCCCCCC'), ('CCCCCCCC', 'TTTTTTTT')])

        # rest should be fairly self explanatory here
        pos, neg, pos_labels, neg_labels = sim_data.simulate(300, 10000)

        train_X = tf.concat([pos, neg], axis=0)
        train_y = tf.concat([pos_labels, neg_labels], axis=0)

        train_y = tf.one_hot(train_y,2)
    # use data from FASTA file
    else:
        seqs = []
        with open(sys.argv[1] + '.fa', 'r') as f:
            i = 1
            for line in f:
                # every other line of a fasta file contains a sequence
                if i % 2 == 0:
                    seqs.append(line[1:].rstrip().upper())
                i += 1
        
        # convert 'N'and 'H' to random base
        DNAalphabet = {'A': '0', 'C': '1', 'G': '2', 'T': '3'}
        for i in range(len(seqs)):
            seqs[i] = seqs[i].replace('N', random.choice(list(DNAalphabet.keys())))
            # NOT QUITE sure what to do with H, not in satori
            seqs[i] = seqs[i].replace('H', random.choice(list(DNAalphabet.keys())))
            seqs[i] = seqs[i].replace('R',list(DNAalphabet.keys())[random.choice([0,2])])
            # Why is there a 4 lol? 
            seqs[i] = seqs[i].replace("4", random.choice(list(DNAalphabet.keys())))
        # convert DNA sequences to vectors of 0,1,2,3
        train_X = tf.convert_to_tensor([map_encode(x) for x in seqs])
        
        # get label information
        bed = pd.read_csv(sys.argv[1] + '.txt', sep = '\t', header = None)
        train_y = tf.convert_to_tensor(bed[3])

    inds = tf.random.shuffle(tf.range(train_X.shape[0]))

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
        