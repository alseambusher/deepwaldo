import os
from Mask_RCNN.mrcnn.config import Config


class Waldoconfig(Config):
    NAME = "waldo"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 2
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9
    COCO_WEIGHTS_PATH = os.path.join("models", "mask_rcnn_coco.h5")
    MODEL_DIR = os.path.join("models", "logs")
    DATA_DIR = "data"

    def __init__(self, predict=False):
        if predict:
            self.IMAGES_PER_GPU = 1
            self.GPU_COUNT = 1
        super(self.__class__, self).__init__();
