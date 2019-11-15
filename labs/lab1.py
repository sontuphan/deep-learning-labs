from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import datasets, models, layers

CHECKPOINT_PATH = "model/lab1.h5"


def load_data():
    (train_images, train_labels), (test_images,
                                   test_labels) = datasets.mnist.load_data()

    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]

    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

    return train_images, train_labels, test_images, test_labels


def create_model():
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model():
    train_images, train_labels, test_images, test_labels = load_data()
    model = create_model()
    model.fit(train_images,
              train_labels,
              epochs=10,
              validation_data=(test_images, test_labels))
    model.save(CHECKPOINT_PATH)


def load_model():
    train_images, train_labels, test_images, test_labels = load_data()
    new_model = models.load_model(CHECKPOINT_PATH)
    loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
    print('Restored model, accuracy: {:5.2f}%'.format(100*acc))


def main():
    # train_model()
    load_model()


if __name__ == "__main__":
    main()
