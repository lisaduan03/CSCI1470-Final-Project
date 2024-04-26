import tensorflow as tf
import pickle as pkl
import numpy as np
from simulator import map_decode

def get_model_outputs(model, test_x, test_y, batch_size=128):
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
        per_batch_testSeqs[batch_idx] = np.array(seqs)  # might need to use np.column_stack(headers,seqs) headers might be region coordinates
        # call model
        x = tf.one_hot(batch_seqs, 4, axis=1)
        x = tf.expand_dims(x, -1)
        x = model.convolutions[0](x)
        per_batch_CNNoutput[batch_idx] = x.numpy()  # might need to instead pickle the output. 
        x = model.convolutions[1](x)
        x = model.convolutions[2](x) # max pool layer
        # x = tf.reshape(x, (-1, 4, model.conv_out_shape))
        x = tf.reshape(x, (-1, 4, (499-20)//5 + 1)) # hardcoded
        x, pAttn_concat = model.self_attns[0](x)
        PAttn_all[batch_idx] = pAttn_concat.numpy()  # might need to instead pickle the attention scores
        x = model.flatten(x)
        for dense in model.dense:
            x = dense(x)
        preds = tf.argmax(x, axis = 1).numpy()
        labels = tf.argmax(batch_labels, axis = 1).numpy()
        label_pred = np.column_stack(labels, preds)
        per_batch_labelPreds[batch_idx] = label_pred

        batch_idx += 1

if __name__ == '__main__':
    # just for testing
    model = tf.keras.models.load_model('saved_model.hd5')
    test_x = pkl.load(open('test_X.pkl', 'rb'))
    test_y = pkl.load(open('test_y.pkl', 'rb'))

    get_model_outputs(model, test_x, test_y)

    print(model.evaluate(test_x, test_y))