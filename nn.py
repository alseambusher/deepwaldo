import tensorflow as tf
import skimage.io
import skimage.transform
import numpy as np
import config
import os
import tensorflow

class WaldoNN:
    def __init__(self):
        pass

    def conv_net(self, rgb, dropout):
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

    def read_images_from_disk(self, input_queue):
        label = input_queue[1]
        file_contents = tf.read_file(input_queue[0])
        example = tf.image.decode_png(file_contents, channels=3)
        example.set_shape([64, 64, 3])
        return example, label

    def get_batch(self):
        train, test = self.get_data()
        image_list = train[:, 0]
        label_list = train[:, 1].astype("i")

        images = tf.convert_to_tensor(image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
        image_batch, label_batch = tf.train.slice_input_producer([images, labels],
                                                    num_epochs=config.n_epochs,
                                                    shuffle=True)
        
        image, label = self.read_images_from_disk((image_batch, label_batch))
       
        # # tf.image implements most of the standard image augmentation
        # # image = preprocess_image(image)
        # # label = preprocess_label(label)
        #
        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size=config.batch_size)

        return image_batch, label_batch, test[:, 0], test[:, 1].astype("i")

    def get_data(self, force_read=False):
        if True or (not os.path.exists(config.trainset_path + ".npy")) or force_read:
            waldo_list = np.array(list(map(lambda x: [os.path.join(config.data_folder, "waldo", x), 1], os.listdir(config.data_folder + "/waldo"))))
            not_waldo_list = np.array(list(map(lambda x: [os.path.join(config.data_folder, "notwaldo", x), 0], os.listdir(config.data_folder + "/notwaldo"))))

            data = np.concatenate([waldo_list, not_waldo_list])
            np.random.shuffle(data)
            train, test = data[:int(data.shape[0]*config.train_split), :], data[int(data.shape[0]*config.train_split):, :]

            np.save(config.trainset_path, train)
            np.save(config.testset_path, test)
            return train[:100], test
        else:
            return np.load(config.trainset_path + ".npy"), np.load(config.testset_path + ".npy")

