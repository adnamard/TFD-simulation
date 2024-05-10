

import numpy as np
import urllib.request
import zipfile
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

def solution_A1():
    # DO NOT CHANGE THIS CODE
    X = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0,
                 2.0, 3.0, 4.0, 5.0], dtype=float)
    Y = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                 12.0, 13.0, 14.0, ], dtype=float)

    # YOUR CODE HERE
    class MyCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') < 1e-4:
          print('\n Training is done, criteria is filled')
          self.model.stop_training = True

    #Sequential

    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    #callback
    my_callback = MyCallback()
    #Compiling
    model.compile(optimizer='sgd', loss='mean_squared_error')
    #Fitting and checking
    model.fit(X, Y, epochs=500, batch_size=8, callbacks=[my_callback])

    print(model.predict([-2.0, 10.0]))
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_A1()
    model.save("model_A1.h5")