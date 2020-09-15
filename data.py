import numpy as np
import tensorflow as tf
import tensorflow.keras as kr

class Dataset:

    def __init__(self, args):

        train_data, test_data = kr.datasets.cifar10.load_data()

        self.batch_size = eval(args['training']['batch_size'])

        self.train_dataset = self._build_dataset(train_data, shuffle=True)
        self.test_dataset  = self._build_dataset(train_data)

    def _build_dataset(self, raw_data, shuffle=False):
        images, labels = raw_data
        labels = np.squeeze(labels).astype(np.int64)[:512]
        images = images.astype(np.float32)[:512]

        # TODO mu, std
        samples = (images / 172.5 - 1., (labels, labels))
        data = tf.data.Dataset.from_tensor_slices(samples)
        if shuffle:
            data = data.shuffle(1000)
        return data.batch(self.batch_size)


