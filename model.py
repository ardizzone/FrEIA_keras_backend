import numpy as np
import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization

from FrEIA.keras.framework import ReversibleSequential
from FrEIA.keras.modules import AllInOneBlock, HaarDownsampling

BatchNormalization._USE_V2_BEHAVIOR = False

class SubnetFactory:
    def __init__(self, ch_hidden):
        self.ch = ch_hidden

    def __call__(self, ch_in, ch_out):
        net = kr.models.Sequential([
                kr.layers.Conv2D(self.ch, 3, padding='same'),
                BatchNormalization(),
                kr.layers.ReLU(),

                kr.layers.Conv2D(self.ch, 3, padding='same'),
                BatchNormalization(),
                kr.layers.ReLU(),

                kr.layers.Conv2D(ch_out, 3, padding='same'),
            ], name='subnet')

        return net


def build_model(args):

    input_data_dims  = eval(args['data']['data_dimensions'])
    n_blocks_per_res = eval(args['model']['inn_coupling_blocks'])
    channels_per_res = eval(args['model']['inn_subnet_channels'])
    clamps_per_res   = eval(args['model']['affine_clamp'])
    actnorm_per_res  = eval(args['model']['global_affine_init'])
    n_res_levels     = len(n_blocks_per_res)


    model = ReversibleSequential(*input_data_dims)
    for l in range(n_res_levels):
        kwargs = {'affine_clamping': clamps_per_res[l],
                  'global_affine_init': actnorm_per_res[l],
                  'global_affine_type': 'SOFTPLUS',
                  'subnet_constructor': SubnetFactory(channels_per_res[l])}

        for k in range(n_blocks_per_res[l]):
            model.append(AllInOneBlock, **kwargs)

        if l < n_res_levels - 1:
            model.append(HaarDownsampling)

    return model


if __name__ == '__main__':
    args = {'data':  {'data_dimensions':     '(32, 32, 3)'},
            'model': {'inn_coupling_blocks': '[2, 4, 4, 4]',
                      'inn_subnet_channels': '[16, 32, 64, 128]',
                      'affine_clamp':        '[1.5] * 4',
                      'global_affine_init':  '[0.8] * 4'}
            }

    test_model = build_model(args)

    x = np.random.randn(16, 32, 32, 3).astype(np.float32)
    x = tf.constant(x)
    z = test_model(x)
