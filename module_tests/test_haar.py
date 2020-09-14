import subnet
from FrEIA.keras.modules.haar import HaarDownsampling as KerasHaarBlock
from module_tests.torch_haar import HaarDownsampling as TorchHaarBlock

import tensorflow.keras as kr
import tensorflow as tf
import torch
import torch.nn
import numpy as np

in_shape = (32, 32, 3)
batch_size = 16

def run_keras_model(x, kwargs):
    block = KerasHaarBlock([in_shape], **kwargs)
    block.output_dims([in_shape])

    x = tf.constant(x)
    inn = kr.Sequential([block])
    z = inn(x)

    return z.numpy()

def run_pytorch_model(x, kwargs):
    in_shape_transp = (in_shape[-1], *in_shape[:-1])
    block = TorchHaarBlock([in_shape_transp], **kwargs)
    block.output_dims([in_shape_transp])

    x = np.transpose(x, (0, 3, 1, 2))
    x = torch.from_numpy(x).contiguous()
    z = block([x])
    z = z[0].data.numpy()
    z = np.transpose(z, (0, 2, 3, 1))
    return z

for k in range(10):
    np.random.seed(k)
    x = np.random.randn(batch_size, *in_shape).astype(np.float32)
    kwargs = {'rebalance' : 3. / (k+1)}

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
