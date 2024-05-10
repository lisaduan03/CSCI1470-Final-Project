import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import pickle as pkl
import os
from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc_curve(true_y, y_prob, label):
    """
    plots the roc curve based on the probabilities
    """
    fpr, tpr, _ = roc_curve(true_y, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.4f})')

# load and test data
def load_data_and_model(directory):
    with open(os.path.join(directory, 'test_X.pkl'), 'rb') as f:
        test_X = pkl.load(f)
    with open(os.path.join(directory, 'test_y.pkl'), 'rb') as f:
        test_y = pkl.load(f)
    model = tf.keras.models.load_model(os.path.join(directory, 'model.hd5'))
    return test_X, test_y, model

# # loop through different number of channels
# channel_sizes = [4, 8, 10, 20, 100]
# for channel_size in channel_sizes:
#     directory = f'real_results/5_attnheads_{channel_size}_channels/'
#     test_X, test_y, model = load_data_and_model(directory)
#     predictions = model.predict(test_X)
#     probabilities = predictions[:, 1]  # Probability of the positive class
#     auc_roc = roc_auc_score(test_y, probabilities)
#     plot_roc_curve(test_y, probabilities, f'{channel_size} channels')

# loop through different attn head sizes
attn_heads = [1, 2, 5, 10]
for attn_head in attn_heads:
    directory = f'real_results/{attn_head}_attnheads_8_channels/'
    test_X, test_y, model = load_data_and_model(directory)
    predictions = model.predict(test_X)
    probabilities = predictions[:, 1]  # Probability of the positive class
    auc_roc = roc_auc_score(test_y, probabilities)
    plot_roc_curve(test_y, probabilities, f'{attn_head} attention heads')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Varying Number of Attention Heads')
plt.legend(loc='lower right')
plt.show()