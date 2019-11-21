from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
# tfds.disable_progress_bar()

import matplotlib.pyplot as plt


def monitor(history):
    history_dict = history.history

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 9))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 9))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim((0.5, 1))
    plt.show()


def main():
    (train_data, test_data), info = tfds.load(
        'imdb_reviews/subwords8k',
        split=(tfds.Split.TRAIN, tfds.Split.TEST),
        with_info=True, as_supervised=True)

    encoder = info.features['text'].encoder
    encoder.subwords[:20]

    padded_shapes = ([None], ())
    train_batches = train_data.shuffle(1000).padded_batch(
        10, padded_shapes=padded_shapes)
    test_batches = test_data.shuffle(1000).padded_batch(
        10, padded_shapes=padded_shapes)

    train_batch, train_labels = next(iter(train_batches))

    embedding_dim = 16

    model = keras.Sequential([
        layers.Embedding(encoder.vocab_size, embedding_dim),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_batches,
        epochs=10,
        validation_data=test_batches, validation_steps=20)

    monitor(history)


if __name__ == "__main__":
    main()
