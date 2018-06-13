import sys
sys.path.append("Mask_RCNN")

from mrcnn.model import MaskRCNN
from waldo_config import Waldoconfig
import numpy as np
import skimage.draw
from PIL import Image

if __name__ == '__main__':

    config = Waldoconfig(predict=True)
    config.display()

    model = MaskRCNN(mode="inference", config=config,
                     model_dir=config.MODEL_DIR)

    weights_path = sys.argv[1]

    print("weights_path: ", weights_path)
    model.load_weights(weights_path, by_name=True)

    image = skimage.io.imread(sys.argv[2])
    masks = model.detect([image], verbose=1)[0]["masks"]

    print("Masks:", masks)

    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    mask_filter = (np.sum(masks, -1, keepdims=True) >= 1)

    if mask_filter.shape[0] > 0:
        waldo = np.where(mask_filter, image, gray).astype(np.uint8)
        img = Image.fromarray(waldo, 'RGB')
        img.show()
    else:
        print("Can't find Waldo. Hmm..")
