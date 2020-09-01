import subnet
import keras_all_in_one_block as aio_block_keras
import torch_all_in_one_block as aio_block_torch

import tensorflow.keras as kr
import tensorflow as tf
import torch
import torch.nn
import numpy as np

in_shape = (32, 32, 3)
batch_size = 16

def run_keras_model(x, kwargs):
    dummy_subnet = kr.models.Sequential([kr.layers.ReLU(negative_slope=1e-2)])
    dummy_subnet_constructor = (lambda cin, cout: dummy_subnet)

    block = aio_block_keras.AllInOneBlock([in_shape], subnet_constructor=dummy_subnet_constructor, **kwargs)

    x = tf.constant(x)
    inn = kr.Sequential([block])
    z = inn(x)

    return z.numpy()

def run_pytorch_model(x, kwargs):
    dummy_subnet = torch.nn.LeakyReLU(negative_slope=1e-2)
    dummy_subnet_constructor = (lambda cin, cout: dummy_subnet)

    block = aio_block_torch.AllInOneBlock([(in_shape[-1], *in_shape[:-1])],
                                          subnet_constructor=dummy_subnet_constructor, **kwargs)

    x = np.transpose(x, (0, 3, 1, 2))
    x = torch.from_numpy(x).contiguous()
    z = block([x])
    z = z[0].data.numpy()
    z = np.transpose(z, (0, 2, 3, 1))
    return z


for k in range(10):
    np.random.seed(k)
    x = np.random.randn(batch_size, *in_shape).astype(np.float32)
    kwargs = {'permutation_random_seed': k,
              'affine_clamping': 0.3,
              'global_affine_type': 'EXP',
             }

    z_keras = run_keras_model(x, kwargs)
    z_pytorch = run_pytorch_model(x, kwargs)

    diff = np.abs(z_keras - z_pytorch)
    diff_rel = diff / np.sqrt(np.abs(z_keras) * np.abs(z_pytorch))

    print('{:8s} {:12s} {:6s}'.format('DIFF ABS', '', ''), end='    ')
    print('{:8s} {:12s} {:6s}'.format('DIFF REL', '', ''))
    for op in [np.max, np.min,  np.mean, np.median]:
        for d in [diff, diff_rel]:
            md = op(d)
            ok = ('OK' if md < 5e-6 else '!!!!')
            print('{:8s} {:12.2E} {:6s}'.format(op.__name__.upper(), md, ok), end='    ')
        print()
    print('------' * 16)
