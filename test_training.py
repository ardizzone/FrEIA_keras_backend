from argparse import ArgumentParser

import tensorflow.keras as kr
import tensorflow as tf
import numpy as np

from FrEIA.keras.framework import ReversibleSequential
from FrEIA.keras.modules import AllInOneBlock

import subnet

#tf.autograph.set_verbosity(3, True)

IN_SHAPE = (32, 32, 3)
NDIMS_TOT = int(np.prod(IN_SHAPE))
BATCH_SIZE = 32

parser = ArgumentParser()
parser.add_argument('--hpdlf', dest='use_hpdlf', action='store_true', default=False)

args = parser.parse_args()

use_hpdlf = args.use_hpdlf
print(use_hpdlf)

if use_hpdlf:
    import tarantella
    tarantella.init(0)
    rank = tarantella.get_rank()
    comm_size = tarantella.get_size()
else:
    rank = 0
    comm_size = 1

print(('RANK:      {}\n'
       'COMM_SIZE: {}').format(rank, comm_size))

model = ReversibleSequential(*IN_SHAPE)

for k in range(5):
    kwargs = {'affine_clamping': 1.0,
              'global_affine_init':0.85,
              'global_affine_type': 'SOFTPLUS',
              'subnet_constructor': subnet.SubnetFactory(64)
             }

    model.append(AllInOneBlock, **kwargs)

def nll_loss_z_part(y, z):
    zz = tf.math.reduce_mean(z**2)
    return 0.5 * zz

def nll_loss_jac_part(y, jac):
    return - tf.math.reduce_mean(jac) / NDIMS_TOT

def _build_dataset(raw_data, shuffle=False):
    images, labels = raw_data
    labels = np.squeeze(labels).astype(np.int64)
    images = images.astype(np.float32)

    samples = (images / 172.5 - 1., (labels, labels))
    data = tf.data.Dataset.from_tensor_slices(samples)
    if shuffle:
        data = data.shuffle(1000)
    return data.batch(BATCH_SIZE)

train_data, test_data = kr.datasets.cifar10.load_data()
train_dataset = _build_dataset(train_data, shuffle=True)
test_dataset  = _build_dataset(train_data)

model.compile(loss=[nll_loss_z_part, nll_loss_jac_part],
              optimizer='adam')

history = model.fit(train_dataset,
                    epochs=4,
                    verbose=True,
                    validation_data=test_dataset)



