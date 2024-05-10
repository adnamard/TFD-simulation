# -*- coding: utf-8 -*-
"""Problem_C2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-zJXZxqiPQCTELthZuVSmnXYccyMFPmW
"""

!pip install tensorflow
!pip install keras

import tensorflow as tf

def solution_C2():
    mnist = tf.keras.datasets.mnist

    class MyCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
        if(logs.get('accuracy') >0.93 and logs.get('val_accuracy') >0.93):
          print('\n Training is done, criteria is filled')
          self.model.stop_training = True

    # NORMALIZE YOUR IMAGE HERE
    (train_image, train_label), (test_image, test_label) = mnist.load_data()

    # NORMALIZE YOUR IMAGE HERE
    train_image = train_image/255
    test_image = test_image/255

    # DEFINE YOUR MODEL HERE
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # End with 10 Neuron Dense, activated by softmax
    #callback
    callbacks = MyCallback()
    # COMPILE MODEL HERE
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # TRAIN YOUR MODEL HERE
    model.fit(train_image, train_label, epochs=20,  verbose=1, validation_data=(test_image, test_label), callbacks=callbacks)
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C2()
    model.save("model_C2.h5")