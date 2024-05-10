# ==========================================================================================================
# PROBLEM A4
#
# Build and train a binary classifier for the IMDB review dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in http://ai.stanford.edu/~amaas/data/sentiment/
#
# Desired accuracy and validation_accuracy > 83%
# ===========================================================================================================

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class MyCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')
        if accuracy > 0.83 and val_accuracy > 0.83:
            print('\n Training is done, criteria is filled')
            self.model.stop_training = True

def solution_A4():
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    # YOUR CODE HERE
    train_data= imdb['train']
    test_data = imdb['test']

    training_sentences = []
    training_labels = []
    testing_sentences = []
    testing_labels = []

    # DO NOT CHANGE THIS CODE
    for s, l in train_data:
        training_sentences.append(s.numpy().decode('utf8'))
        training_labels.append(l.numpy())

    for s, l in test_data:
        testing_sentences.append(s.numpy().decode('utf8'))
        testing_labels.append(l.numpy())

    # YOUR CODE HERE

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size =10000
    max_length =120
    embedding_dim =16
    trunc_type ='post'
    padding_type='post'
    oov_tok ="<OOV>"

    # Fit your tokenizer with training data
    tokenizer =  Tokenizer(num_words = 10000, oov_token=oov_tok)

    #fitting
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    # Generate and pad the training sequences
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Generate and pad the testing sequences
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    training_labels = np.array(training_labels)
    testing_labels = np.array(testing_labels)
    #params
    embedding_dim = 16
    dense_dim = 32

    model = tf.keras.Sequential([
        # YOUR CODE HERE. Do not change the last layer.
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    #compiling
    model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),metrics=['accuracy'])

    #callback
    callbacks = MyCallbacks()

    #training model
    sum_epochs = 50
    model.fit(training_padded, training_labels, epochs=sum_epochs, validation_data=(testing_padded, testing_labels), callbacks=[callbacks])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_A4()
    model.save("model_A4.h5")