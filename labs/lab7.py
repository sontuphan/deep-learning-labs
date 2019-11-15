import os

from PIL import Image, ImageDraw
import numpy as np
import cv2 as cv

import camera
import detect
import tflite_runtime.interpreter as tflite


EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
LABELS = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../model/coco_labels.txt")
MODELS = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../model/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")
IMAGE_SHAPE = (150, 150)


def load_labels():
    with open(LABELS, 'r', encoding='utf-8') as labels_file:
        lines = labels_file.readlines()
        if not lines:
            return {}

        pairs = [line.split(' ', maxsplit=1) for line in lines]
        return {int(index): label.strip() for index, label in pairs}


def load_image(stream):
    img = stream.get()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil


def make_interpreter():
    return tflite.Interpreter(
        model_path=MODELS,
        experimental_delegates=[
            tflite.load_delegate(EDGETPU_SHARED_LIB)
        ])


def draw_objects(draw, objs, labels):
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                  fill='red')


def main():
    labels = load_labels()
    interpreter = make_interpreter()
    interpreter.allocate_tensors()

    cam = camera.Camera()
    stream = cam.get_stream()
    print("you can press Q button to terminate the process!")

    while True:
        image = load_image(stream)
        scale = detect.set_input(interpreter, image.size,
                                 lambda size: image.resize(size, Image.ANTIALIAS))

        interpreter.invoke()
        objs = detect.get_output(interpreter, 0.4, scale)
        draw_objects(ImageDraw.Draw(image), objs, labels)

        cv.imshow("Debug", np.asarray(image))
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cam.terminate()


if __name__ == '__main__':
    main()
