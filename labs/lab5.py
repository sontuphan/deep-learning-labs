from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

tfds.disable_progress_bar()


IMG_SIZE = 160  # All images will be resized to 160x160
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
SPLIT_WEIGHTS = (8, 1, 1)


def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


def load_data():
    splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

    (raw_train, raw_validation, raw_test), metadata = tfds.load(
        'cats_vs_dogs', split=list(splits),
        with_info=True, as_supervised=True)

    train = raw_train.map(format_example)
    validation = raw_validation.map(format_example)
    test = raw_test.map(format_example)

    train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    validation_batches = validation.batch(BATCH_SIZE)
    test_batches = test.batch(BATCH_SIZE)

    return train_batches, validation_batches, test_batches, metadata


def create_model():
    base_model = keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
    base_model.trainable = False

    global_average_layer = keras.layers.GlobalAveragePooling2D()
    prediction_layer = keras.layers.Dense(1)

    model = keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def monitor(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def main():
    train_batches, validation_batches, test_batches, metadata = load_data()
    model = create_model()

    num_train, num_val, num_test = (
        metadata.splits['train'].num_examples*weight/10
        for weight in SPLIT_WEIGHTS
    )

    initial_epochs = 10
    steps_per_epoch = round(num_train)//BATCH_SIZE
    validation_steps = 20

    history = model.fit(train_batches,
                        epochs=initial_epochs,
                        validation_data=validation_batches)

    monitor(history)


if __name__ == "__main__":
    main()
