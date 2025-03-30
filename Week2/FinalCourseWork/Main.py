import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from public_tests import *
from autils import *
from lab_utils_softmax import plt_softmax
np.set_printoptions(precision=2)

def my_softmax(z):
    e_z = np.exp(z)
    a = e_z/np.sum(e_z)

    return a

# load dataset
X, y = load_data()

print()

# UNQ_C2
# GRADED CELL: Sequential model
tf.random.set_seed(1234)  # for consistent results
model = Sequential(
    [
        ### START CODE HERE ###
        Dense(25, activation='relu', input_shape=(400,)),
        Dense(15, activation='relu'),
        Dense(10)
        ### END CODE HERE ###
    ], name="my_model"
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

history = model.fit(
    X,y,
    epochs=40
)
