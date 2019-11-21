import os

from PIL import Image, ImageDraw
import numpy as np
import cv2 as cv

import tflite_runtime.interpreter as tflite

import camera
import detect


class ObjectRecognition():
    def __init__(self):
        self.LABELS = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "../model/coco_labels.txt")
        self.MODELS = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "../model/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")
        self.IMAGE_SHAPE = (150, 150)

        self.thresold = 10
        self.labels = self.load_labels()
        self.interpreter = tflite.Interpreter(
            model_path=self.MODELS,
            experimental_delegates=[
                tflite.load_delegate("libedgetpu.so.1")
            ])
        self.interpreter.allocate_tensors()

    def load_labels(self):
        with open(self.LABELS, 'r', encoding='utf-8') as labels_file:
            lines = labels_file.readlines()
            if not lines:
                return {}

            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}

    def load_image(self, stream):
        img = stream.get()
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        return im_pil

    def draw_objects(self, draw, objs, labels):
        for obj in objs:
            bbox = obj.bbox
            (cx, cy) = (int((bbox.xmin + bbox.xmax)/2),
                        int((bbox.ymin + bbox.ymax)/2))
            draw.ellipse((cx-3, cy-3, cx+3, cy+3), fill='red', outline='red')
            draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                           outline='red')
            draw.text((bbox.xmin + 10, bbox.ymin + 10),
                      '%s\n%.2f' % (
                          labels.get(obj.id, obj.id), obj.score),
                      fill='red')

    def track(self):
        cam = camera.Camera()
        stream = cam.get_stream()
        print("You can press Q button to terminate the process!")

        while True:
            image = self.load_image(stream)
            scale = detect.set_input(self.interpreter, image.size,
                                     lambda size: image.resize(size, Image.ANTIALIAS))

            self.interpreter.invoke()
            objs = detect.get_output(self.interpreter, 0.4, scale)
            self.draw_objects(ImageDraw.Draw(image), objs, self.labels)

            cv.imshow("Debug", np.asarray(image))
            if cv.waitKey(10) & 0xFF == ord('q'):
                break

        cam.terminate()

    def detect(self):
        cam = camera.Camera()
        stream = cam.get_stream()
        print("You can press Q button to terminate the process!")

        while True:
            image = self.load_image(stream)
            scale = detect.set_input(self.interpreter, image.size,
                                     lambda size: image.resize(size, Image.ANTIALIAS))

            self.interpreter.invoke()
            objs = detect.get_output(self.interpreter, 0.4, scale)
            self.draw_objects(ImageDraw.Draw(image), objs, self.labels)

            cv.imshow("Debug", np.asarray(image))
            if cv.waitKey(10) & 0xFF == ord('q'):
                break

        cam.terminate()


if __name__ == '__main__':
    model = ObjectRecognition()
    model.detect()
    # model.track()
