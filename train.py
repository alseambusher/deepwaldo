import os
import sys
sys.path.append("Mask_RCNN")

from mrcnn.model import MaskRCNN
from mrcnn import utils
from data import Data
from waldo_config import Waldoconfig

if __name__ == '__main__':
    config = Waldoconfig()
    config.display()

    model = MaskRCNN(mode="training", config=config,
                              model_dir=config.MODEL_DIR)

    weights_path = config.COCO_WEIGHTS_PATH
    # Download weights file
    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)

    # don't load last layers because we are going to train it
    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    dataset_train = Data()
    dataset_train.load(config.DATA_DIR, "train")
    dataset_train.prepare()

    dataset_val = Data()
    dataset_val.load(config.DATA_DIR, "val")
    dataset_val.prepare()

    # training just heads layer is enough
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')
