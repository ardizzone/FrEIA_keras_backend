from collections import namedtuple

import subnet
from keras_all_in_one_block import AllInOneBlock
from keras_reversible_sequential_net import ReversibleSequential

import tensorflow.keras as kr
import tensorflow as tf
import torch
import torch.nn
import numpy as np

in_shape = (32, 32, 3)
ndims_tot = int(np.prod(in_shape))
BATCH_SIZE = 32

model = ReversibleSequential(*in_shape)

for k in range(5):
    kwargs = {'affine_clamping': 1.0,
              'global_affine_init':0.7,
              'global_affine_type': 'SOFTPLUS',
              'subnet_constructor': subnet.SubnetFactory(64)
             }

    model.append(AllInOneBlock, **kwargs)

def nll_loss_z_part(y, z):
    zz = tf.math.reduce_mean(z**2)
    return 0.5 * zz

def nll_loss_jac_part(y, jac):
    return - tf.math.reduce_mean(jac) / ndims_tot

Sample = namedtuple('Sample', ['image', 'label'])

def _build_dataset(raw_data, shuffle=False):
    images, labels = raw_data
    labels = np.squeeze(labels)
    samples = Sample(images.astype(np.float32) / 172.5 - 1., labels.astype(np.int64))
    data = tf.data.Dataset.from_tensor_slices(samples)
    if shuffle:
        data = data.shuffle(1000)
    return data.batch(BATCH_SIZE)#.make_one_shot_iterator().get_next()

train_data, test_data = kr.datasets.cifar10.load_data()
train_dataset = _build_dataset(train_data, shuffle=True)
test_dataset  = _build_dataset(train_data)

model.compile(loss=[nll_loss_z_part, nll_loss_jac_part],
              optimizer='adam')

history = model.fit(train_dataset,
                    epochs=12,
                    verbose=True,
                    validation_data=test_dataset)



