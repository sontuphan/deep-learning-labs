from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

import matplotlib.pylab as plt
import numpy as np
import PIL.Image as Image

MODEL = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
# MODEL = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
DATA = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../images/cats.jpg")
IMAGE_SHAPE = (150, 150)


def main():
    cats = Image.open(DATA).resize(IMAGE_SHAPE)
    cats = np.array(cats)/255.0
    print(cats.shape)
    tensors = cats[np.newaxis, ...]
    print(tensors.shape)

    detector = hub.load(MODEL).signatures["default"]
    detector_output = detector(tensors)
    class_names = detector_output["detection_class_names"]
    print(class_names)


if __name__ == "__main__":
    main()
