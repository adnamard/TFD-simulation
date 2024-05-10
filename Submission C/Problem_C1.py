# -*- coding: utf-8 -*-
"""Problem_C1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1L3i8VO0MdMgMGK9LsQukhUowChsZNtHD
"""

!pip install tensorflow
!pip install keras
!pip install keras-preprocessing

import numpy as np
import tensorflow as tf
from tensorflow import keras

# =============================================================================
# PROBLEM C1
#
# Given two arrays, train a neural network model to match the X to the Y.
# Predict the model with new values of X [-2.0, 10.0]
# We provide the model prediction, do not change the code.
#
# The test infrastructure expects a trained model that accepts
# an input shape of [1]
# Do not use lambda layers in your model.
#
# Please be aware that this is a linear model.
# We will test your model with values in a range as defined in the array to make sure your model is linear.
#
# Desired loss (MSE) < 1e-4
# =============================================================================

def solution_C1():
    # DO NOT CHANGE THIS CODE
    X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    Y = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

    # YOUR CODE HERE

    class MyCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
        if(logs.get('loss') < 1e-4):
          print('\n Training is done, criteria is filled')
          self.model.stop_training = True

    #normalization
    normalizer = tf.keras.layers.Normalization(axis=None, input_shape=(1,))
    normalizer.adapt(X)

    #seq and modelling
    model = keras.Sequential()
    model.add(normalizer)
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dense(1))

    #callback
    callback= MyCallback()

    #compillling and fitting
    model.compile(loss='mse', optimizer='sgd')
    model.fit(X, Y, epochs=1000, callbacks=callback)

    print(model.predict([-2.0, 10.0]))
    return model


# The code below is to save your model as a .h5 file
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C1()
    model.save("model_C1.h5")