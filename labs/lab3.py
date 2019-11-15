from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 128
EPOCHS = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150


def load_data():

    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    path_to_zip = tf.keras.utils.get_file(
        'cats_and_dogs.zip', origin=_URL, extract=True)
    PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

    print("Data dir:", PATH)
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')
    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')

    num_cats_tr = len(os.listdir(train_cats_dir))
    num_dogs_tr = len(os.listdir(train_dogs_dir))
    num_cats_val = len(os.listdir(validation_cats_dir))
    num_dogs_val = len(os.listdir(validation_dogs_dir))

    total_train = num_cats_tr + num_dogs_tr
    total_val = num_cats_val + num_dogs_val
    info = [total_train, total_val]

    train_image_generator = image.ImageDataGenerator(
        rescale=1./255,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        rotation_range=45,
        zoom_range=0.5)
    train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(
                                                                   IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='binary')

    validation_image_generator = image.ImageDataGenerator(rescale=1./255)
    val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                                  directory=validation_dir,
                                                                  target_size=(
                                                                      IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='binary')

    return train_data_gen, val_data_gen, info


def view_data(train_data_gen):
    sample_training_images, _ = next(train_data_gen)
    images_arr = sample_training_images[:5]
    plotImages(images_arr)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def create_model():
    model = models.Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu',
                      input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def save_model(model):
    model.save("model/lab3.h5")


def load_model():
    return models.load_model("model/lab3.h5")


def monitor(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def main():
    train_data_gen, val_data_gen, [total_train, total_val] = load_data()
    # view_data(train_data_gen)

    model = create_model()
    model.summary()
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps=total_val // BATCH_SIZE
    )
    save_model(model)
    monitor(history)


if __name__ == "__main__":
    main()
