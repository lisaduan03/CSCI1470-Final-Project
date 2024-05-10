import tensorflow as tf
import pickle as pkl
import numpy as np
# from simulator import map_decode

def get_model_outputs(model, test_x, test_y, batch_size=128, num_heads=2, seq_len=300):
    roc = np.asarray([[],[]]).T
    per_batch_labelPreds = {}
    per_batch_testSeqs = {}
    per_batch_CNNoutput = {}
    PAttn_all = {}
    num_samples = test_x.shape[0]
    batch_idx = 0
    for i in range(batch_size, num_samples, batch_size):
        batch_seqs = test_x[i-batch_size:i]
        batch_labels = test_y[i-batch_size:i]
        seqs = [map_decode(seq) for seq in batch_seqs.numpy()]
        # generate random header labels
        headers = []
        num_seqs = len(seqs)
        for i in range(num_seqs):
            start = np.random.randint(1000)
            end = np.random.randint(1000)
            headers.append('>chrZ:' + str(start) + '-' + str(end) + '(.)')
        per_batch_testSeqs[batch_idx] = np.column_stack((headers,seqs))  # might need to use np.column_stack(headers,seqs) headers are probably region coordinates. 
        # call model
        x = tf.one_hot(batch_seqs, 4, axis=1)
        x = tf.expand_dims(x, -1)
        x = model.convolutions[0](x) 
        per_batch_CNNoutput[batch_idx] = tf.transpose(tf.reshape(x, (x.shape[0], x.shape[2], x.shape[3])), (0,2,1)).numpy()
        x = model.convolutions[1](x) # max pool layer
        conv_out_shape = (seq_len - 30)//5 + 1 # hardcoded
        x = tf.reshape(x, (-1, conv_out_shape, 8))
        x = x + positional_encoding(conv_out_shape, 8)
        x, pAttn_concat = model.self_attns[0](x)

        # attention_scores_list = []
        # for i in range(0,batch_size*num_heads,batch_size):
        #     attention_scores_list.append(pAttn_concat[i:i+batch_size,:,:])

        # print(pAttn_concat[:128,:,:].shape)
        # print(pAttn_concat[128:,:,:].shape)
        # pAttn_concat = np.stack([pAttn_concat[:128,:,:], pAttn_concat[128:,:,:]], axis=3)
        # pAttn_concat = np.stack(attention_scores_list, axis=3)
        # pAttn_concat = np.reshape(pAttn_concat, (batch_size, conv_out_shape, 2*conv_out_shape))
        print(pAttn_concat.shape)
        PAttn_all[batch_idx] = pAttn_concat  # might need to instead pickle the attention scores
        x = model.flatten(x)
        for dense in model.dense:
            x = dense(x)
        preds = tf.argmax(x, axis = 1).numpy()
        labels = tf.argmax(batch_labels, axis = 1).numpy()
        label_pred = np.column_stack((labels, preds))
        per_batch_labelPreds[batch_idx] = label_pred
        roc = np.row_stack((roc,label_pred))
        batch_idx += 1

    loss, valid_auc, _ = model.evaluate(test_x, test_y)

    print('num batches: ')
    print(batch_idx)

    return loss, valid_auc, roc, PAttn_all, per_batch_labelPreds, per_batch_CNNoutput, per_batch_testSeqs

# for categorical encoding of sequences (prior to OHE)
MAP_ENCODE = {'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3}
MAP_DECODE = {0 : 'A', 1 : 'C', 2 : 'G', 3 : 'T'}


def map_encode(seq : str):
    '''
    Turn sequence in {A, C, G, T}^+ into a list of {0, 1, 2, 3}^+.

    Parameters:
    -----------
    seq - sequence of {A, C, G, T} as a string

    Return:
    -------
    A list of {0, 1, 2, 3}
    '''
    split_seq = [*seq]
    return [MAP_ENCODE[x] for x in split_seq]

def map_decode(seq : list[int]):
    '''
    Turn a list sequence in {0, 1, 2, 3}^+ into a string in {A, C, G, T}^+.

    Parameters:
    -----------
    seq - a list of {0, 1, 2, 3}

    Return:
    -------
    A sequence of {A, C, G, T} as a string
    '''
    split_seq = [MAP_DECODE[x] for x in seq]
    return ''.join(split_seq)

def positional_encoding(length, depth):
    ## REFERENCE: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
    ## TODO: Can remove signature
    depth = depth/2
    ## Generate a range of positions and depths 
    positions = np.arange(length)[:, np.newaxis]    # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth  # (1, depth)
    ## Compute range of radians to take the sine and cosine of.
    angle_rates = 1 / (10000**depths)               # (1, depth)
    angle_rads = positions * angle_rates            # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    ## This serves as offset for the Positional Encoding
    return tf.cast(pos_encoding, dtype=tf.float32)

if __name__ == '__main__':
    # just for testing
    model = tf.keras.models.load_model('model_promoters_nolstm.hd5')
    test_x = pkl.load(open('test_X.pkl', 'rb'))
    test_y = pkl.load(open('test_y.pkl', 'rb'))

    get_model_outputs(model, test_x, test_y)
