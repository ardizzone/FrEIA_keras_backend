import numpy as np
import tensorflow as tf
import tensorflow.keras as kr

class Dataset:

    def __init__(self, args):

        train_data, test_data = kr.datasets.cifar10.load_data()

        self.batch_size = eval(args['training']['batch_size'])

        self.mu = eval(args['data']['mu_normalize'])
        self.std = eval(args['data']['std_normalize'])

        self.mu =  np.reshape(self.mu,  (1, 1, 1, 3)).astype(np.float32)
        self.std = np.reshape(self.std, (1, 1, 1, 3)).astype(np.float32)

        self.train_dataset = self._build_dataset(train_data, shuffle=True)
        self.test_dataset  = self._build_dataset(train_data, shuffle=False, cut=512)


    def _build_dataset(self, raw_data, shuffle=False, cut=None):
        images, labels = raw_data
        if cut:
            images = images[:cut]
            labels = labels[:cut]

        labels = np.squeeze(labels).astype(np.int64)

        images = images.astype(np.float32)
        images /= 255.
        images = images - self.mu
        images = images / self.std

        samples = (images, (labels, labels))

        data = tf.data.Dataset.from_tensor_slices(samples)
        if shuffle:
            data = data.shuffle(1000)
        return data.batch(self.batch_size)


