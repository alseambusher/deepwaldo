import skimage.io
import numpy as np

class Predict:
    def __init__(self, session, path):
        self.session = session
        self.image = skimage.io.imread(path).astype(float)

    def find_waldo(self, prediction, images_tf, keep_prob):
        height, width, channels = self.image.shape
        # TODO try with a different stride
        parts = []
        for i in range(height-64+1):
            for j in range(width-64+1):
                parts.append(self.image[i:i+64,j:j+64, :])
        print(len(parts))
        # TODO batching 
        pred = self.session.run([prediction], feed_dict={images_tf: np.array(parts[:1000]),
                                        keep_prob: 1.0})
        print(pred)


