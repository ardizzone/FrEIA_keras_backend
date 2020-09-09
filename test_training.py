import subnet
from keras_all_in_one_block import AllInOneBlock
from keras_reversible_sequential_net import ReversibleSequential

import tensorflow.keras as kr
import tensorflow as tf
import torch
import torch.nn
import numpy as np

in_shape = (32, 32, 3)
batch_size = 16

net = ReversibleSequential(*in_shape)

def subnet_constr(cin, cout):
    return kr.models.Sequential([
            kr.layers.Conv2D(64, 3, padding='same'),
            kr.layers.BatchNormalization(),
            kr.layers.ReLU(),

            kr.layers.Conv2D(64, 3, padding='same'),
            kr.layers.BatchNormalization(),
            kr.layers.ReLU(),

            kr.layers.Conv2D(cout, 3, padding='same'),
        ])

for k in range(5):
    kwargs = {'affine_clamping': 1.0,
              'global_affine_init':0.7,
              'global_affine_type': 'SOFTPLUS',
              'subnet_constructor': subnet_constr
             }

    net.append(AllInOneBlock, **kwargs)

x = np.random.randn(batch_size, *in_shape).astype(np.float32)
x = tf.constant(x)
z, jac = net(x)
z, jac = z.numpy(), jac.numpy()

print('mean(z)  ', np.mean(z))
print('std(z)   ', np.std(z))
print('mean(jac)', np.mean(jac))
print('std(jac) ', np.std(jac))
print('spread jac', np.min(jac) - np.max(jac))
