import ast
import os
import io
import logging
from collections import namedtuple

import numpy as np
import cv2
from PIL import Image

from gabriel_protocol import gabriel_pb2
from gabriel_server import local_engine
from gabriel_server import cognitive_engine

import tensorflow as tf
from object_detection.utils import label_map_util

import torch
from torchvision import transforms

import mpncov

SOURCE = 'profiling'
INPUT_QUEUE_MAXSIZE = 60
PORT = 9099
NUM_TOKENS = 1
DETECTOR_ONES_SIZE = (1, 480, 640, 3)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

LABELS_FILENAME = 'classes.txt'
CLASSIFIER_FILENAME = 'model_best.pth.tar'
LABEL_MAP_FILENAME = 'label_map.pbtxt'
DETECTOR_CLASS_NAME = "default"
CONF_THRESHOLD = 0.4

_Classifier = namedtuple('_Classifier', ['model', 'labels'])
_Detector = namedtuple('_Detector', ['detector', 'category_index'])


class InferenceEngine(cognitive_engine.Engine):
    def __init__(self):
        self.input_count = 0

        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self._transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            normalize,
        ])

        self._classifier_representation = {
            'function': mpncov.MPNCOV,
            'iterNum': 5,
            'is_sqrt': True,
            'is_vec': True,
            'input_dim': 2048,
            'dimension_reduction': None,
        }

        classifier_dir = os.path.join(MODELS_DIR, "classifier")
        labels_file = open(os.path.join(classifier_dir, LABELS_FILENAME))
        labels = ast.literal_eval(labels_file.read())
        freezed_layer = 0
        model = mpncov.Newmodel(self._classifier_representation.copy(),
                                len(labels), freezed_layer)
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        trained_model = torch.load(os.path.join(classifier_dir, CLASSIFIER_FILENAME))
        model.load_state_dict(trained_model['state_dict'])
        model.eval()
        self._classifier = _Classifier(model=model, labels=labels)

        detector_dir = os.path.join(MODELS_DIR, "detector")
        detector = tf.saved_model.load(detector_dir)
        ones = tf.ones(DETECTOR_ONES_SIZE, dtype=tf.uint8)
        detector(ones)
        label_map_path = os.path.join(detector_dir, LABEL_MAP_FILENAME)
        label_map = label_map_util.load_labelmap(label_map_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=label_map_util.get_max_label_map_index(label_map),
            use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        self._detector = _Detector(detector=detector, category_index=category_index)

    def handle(self, input_frame):
        np_data = np.frombuffer(input_frame.payloads[0], dtype=np.uint8)
        img_bgr = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        detections = self._detector.detector(np.expand_dims(img, 0))
        scores = detections['detection_scores'][0].numpy()
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(int)

        im_height, im_width = img.shape[:2]
        pil_img = Image.open(io.BytesIO(input_frame.payloads[0]))

        good_boxes = []
        box_scores = []
        for score, box, class_id in zip(scores, boxes, classes):
            class_name = self._detector.category_index[class_id]['name']
            if score > CONF_THRESHOLD and class_name == DETECTOR_CLASS_NAME:
                bi = 0
                while bi < len(box_scores):
                    if score > box_scores[bi]:
                        break
                    bi += 1
                good_boxes.insert(bi, box)
                box_scores.insert(bi, score)

        print()
        print('Detector boxes:', box_scores)

        if good_boxes:
            best_box = good_boxes[0]
            ymin, xmin, ymax, xmax = best_box
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            cropped_pil = pil_img.crop((left, top, right, bottom))
            transformed = self._transform(cropped_pil).cuda()
            output = self._classifier.model(transformed[None, ...])
            prob = torch.nn.functional.softmax(output, dim=1)
            print('Classifier probability:', prob.data.cpu().numpy())

            value, pred = prob.topk(1, 1, True, True)
            class_ind = pred.item()
            label_name = self._classifier.labels[class_ind]
            logger.info('Found label: %s', label_name)

        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)

        # result = gabriel_pb2.ResultWrapper.Result()
        # result.payload_type = gabriel_pb2.PayloadType.IMAGE
        # result.payload = input_frame.payloads[0]
        # result_wrapper.results.append(result)
        self.input_count += 1
        logger.info("Input count: {}".format(self.input_count))

        return result_wrapper


def main():
    def engine_factory():
        return InferenceEngine()

    local_engine.run(engine_factory, SOURCE, INPUT_QUEUE_MAXSIZE,
                     PORT, NUM_TOKENS)


if __name__ == '__main__':
    main()
