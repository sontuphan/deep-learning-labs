from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers

import numpy as np
import PIL.Image as Image


def plotImages(image_batch, predicted_class_names):
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)
    for n in range(30):
        plt.subplot(6, 5, n+1)
        plt.imshow(image_batch[n])
        plt.title(predicted_class_names[n])
        plt.axis('off')
    _ = plt.suptitle("ImageNet predictions")
    plt.show()


def main():
    classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
    IMAGE_SHAPE = (224, 224)

    classifier = tf.keras.Sequential([
        hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
    ])

    data_root = tf.keras.utils.get_file(
        'flower_photos', 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        untar=True)
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255)
    image_data = image_generator.flow_from_directory(
        str(data_root), target_size=IMAGE_SHAPE)

    for image_batch, label_batch in image_data:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break

    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    result_batch = classifier.predict(image_batch)
    predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
    plotImages(image_batch, predicted_class_names)


if __name__ == "__main__":
    main()
