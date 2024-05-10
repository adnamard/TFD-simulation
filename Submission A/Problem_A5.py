

import csv
import tensorflow as tf
import numpy as np
import urllib

class MyCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mae = logs.get('mae')
        if mae < 0.15:
            print('\n Training is done, criteria is filled')
            self.model.stop_training = True


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def solution_A5():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sunspots.csv'
    urllib.request.urlretrieve(data_url, 'sunspots.csv')

    time_step = []
    sunspots = []

    with open('sunspots.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
        sunspots.append(int(row[0]))# YOUR CODE HERE)
        time_step.append(float(row[2]))# YOUR CODE HERE)

    series=np.array(sunspots)# YOUR CODE HERE

    # Normalization Function. DO NOT CHANGE THIS CODE
    min=np.min(series).astype(np.float64)
    max=np.max(series).astype(np.float64)
    time=np.array(time_step)
    series = (series - min) / (max- min)

    # DO NOT CHANGE THIS CODE
    split_time=3000


    time_train=time[:split_time]# YOUR CODE HERE
    x_train=series[:split_time]# YOUR CODE HERE
    time_valid=time[split_time:]# YOUR CODE HERE
    x_valid=series[split_time:]# YOUR CODE HERE

    # DO NOT CHANGE THIS CODE
    window_size=30
    batch_size=32
    shuffle_buffer_size=1000


    train_set=windowed_dataset(x_train, window_size=window_size,
                               batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)


    model=tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                               strides=1,
                               activation="relu",
                               padding='causal',
                               input_shape=[None, 1]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, dropout=0.2),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    #set params
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    #callback
    callbacks = MyCallbacks()

    #training
    model.fit(train_set,epochs=120, callbacks=[callbacks])
    # YOUR CODE

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_A5()
    model.save("model_A5.h5")