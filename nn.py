import tensorflow as tf
import skimage.io
import skimage.transform
import numpy as np
import config
import os


class WaldoNN:
    def __init__(self):
        pass

    def conv_net(self, rgb, dropout):
        # r, g, b = tf.split(3, 3, rgb)
        # # TODO Get it to 0 mean
        # bgr = tf.concat(3,
        #                 [
        #                     b,
        #                     g,
        #                     r
        #                 ])
        network = tf.contrib.layers.conv2d(rgb, 64, 3)
        network = tf.contrib.layers.max_pool2d(network, 2, stride=1, padding="SAME")
        network = tf.contrib.layers.conv2d(network, 64, 3)
        network = tf.contrib.layers.conv2d(network, 64, 3)
        network = tf.contrib.layers.max_pool2d(network, 2, stride=1, padding="SAME")

        network = tf.contrib.layers.flatten(network)
        network = tf.contrib.layers.fully_connected(network, 512)

        # don't apply dropout if not in training
        network = tf.contrib.layers.dropout(network, keep_prob=dropout)
        # output layer for class prediction
        network = tf.contrib.layers.fully_connected(network, 2, activation_fn=tf.nn.softmax)
        return network

    @staticmethod
    def load_image(path):
        try:
            return skimage.io.imread(path).astype(float)
        except:
            print("Image not found", path)
            return None

    @staticmethod
    def get_data(force_read=False):
        if (not os.path.exists(config.trainset_path + ".npy")) or force_read:
            waldo_list = np.array(list(map(lambda x: [os.path.join(config.data_folder, "waldo", x), 1], os.listdir(config.data_folder + "/waldo"))))
            not_waldo_list = np.array(list(map(lambda x: [os.path.join(config.data_folder, "notwaldo", x), 0], os.listdir(config.data_folder + "/notwaldo"))))

            data = np.concatenate([waldo_list, not_waldo_list])
            np.random.shuffle(data)
            train, test = data[:int(data.shape[0]*config.train_split), :], data[int(data.shape[0]*config.train_split):, :]

            np.save(config.trainset_path, train)
            np.save(config.testset_path, test)
            return train, test
        else:
            return np.load(config.trainset_path + ".npy"), np.load(config.testset_path + ".npy")

