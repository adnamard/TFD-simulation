# -*- coding: utf-8 -*-
"""Problem_C5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mTys1gPjwyo3eRSxIua3hUYRbgwjwgLs
"""

!pip install tensorflow
!pip install keras
!pip install keras-preprocessing

import urllib
import os
import zipfile
import pandas as pd
import tensorflow as tf

class MyCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('mean_absolute_error')<0.09 and logs.get('val_mean_absolute_error')<0.09):
            print('\n Training is done, val is filled')
            self.model.stop_training = True

def download_and_extract_data():
    url = 'https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/household_power.zip'
    urllib.request.urlretrieve(url, 'household_power.zip')
    with zipfile.ZipFile('household_power.zip', 'r') as zip_ref:
        zip_ref.extractall()


# This function normalizes the dataset using min max scaling.
# DO NOT CHANGE THIS CODE
def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data

# COMPLETE THE CODE IN THE FOLLOWING FUNCTION.
def windowed_dataset(series, batch_size, n_past=24, n_future=24, shift=1):
    # YOUR CODE HERE
    datas = tf.data.Dataset.from_tensor_slices(series)
    datas = datas.window(n_past + n_future, shift=shift, drop_remainder=True)
    datas = datas.flat_map(lambda window: window.batch(n_past + n_future))
    datas =datas.shuffle(1000)
    datas = datas.map(lambda window: (window[:-n_future], window[-n_future:, :1]))
    datas = datas.batch(batch_size).prefetch(1)
    return datas
     # YOUR CODE HERE

# COMPLETE THE CODE IN THE FOLLOWING FUNCTION.
def solution_C5():
    # Downloads and extracts the dataset to the directory that contains this file.
    download_and_extract_data()
    # Reads the dataset from the csv.
    df = pd.read_csv('household_power_consumption.csv', sep=',',
                     infer_datetime_format=True, index_col='datetime', header=0)

    # Number of features in the dataset. We use all features as predictors to
    # predict all features at future time steps.
    N_FEATURES = 7# YOUR CODE HERE

    # Normalizes the data
    # DO NOT CHANGE THIS
    data = df.values
    split_time = int(len(data) * 0.5)
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    # Splits the data into training and validation sets.
    x_train = data[:split_time:]# YOUR CODE HERE
    x_valid = data[split_time:]# YOUR CODE HERE

    # DO NOT CHANGE THIS
    BATCH_SIZE = 32
    N_PAST = 24 # Number of past time steps based on which future observations should be predicted
    N_FUTURE = 24  # Number of future time steps which are to be predicted.
    SHIFT = 1  # By how many positions the window slides to create a new window of observations.

    # Code to create windowed train and validation datasets.
    # Complete the code in windowed_dataset.
    train_set = windowed_dataset(x_train, BATCH_SIZE, N_PAST, N_FUTURE, SHIFT)# YOUR CODE HERE
    valid_set = windowed_dataset(x_valid, BATCH_SIZE, N_PAST, N_FUTURE, SHIFT)# YOUR CODE HERE

    # Code to define your model.
    model = tf.keras.models.Sequential([
        # Whatever your first layer is, the input shape will be (N_PAST = 24, N_FEATURES = 7)
        tf.keras.layers.Conv1D(64, kernel_size=5, strides=1, padding="causal", input_shape=(N_PAST, N_FEATURES), activation='relu'),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        # YOUR CODE HERE
        tf.keras.layers.Dense(N_FUTURE),
    ])

    # Code to train and compile the model
    # YOUR CODE HERE
    #compilling
    model.compile(optimizer='adam', loss='mae', metrics=['mean_absolute_error'])

    #callback
    callbacks = MyCallbacks()

    #fitting
    model.fit(train_set, verbose=1, epochs=15, validation_data=valid_set, callbacks=callbacks)
    return model

# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C5()
    model.save("model_C5.h5")