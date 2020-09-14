import numpy as np
import tensorflow as tf
import tensorflow.keras as kr

class HaarDownsampling(kr.layers.Layer):
    '''Uses Haar wavelets to split each channel into 4 channels, with half the
    width and height.'''

    def __init__(self, dims_in, order_by_wavelet=False, rebalance=1.):
        super().__init__()

        self.in_channels = dims_in[0][-1]
        self.fac_fwd = 0.5 * rebalance
        self.fac_rev = 0.5 / rebalance
        self.haar_weights = np.ones((2,2,1,4)).astype(np.float32)

        self.haar_weights[0, 1, 0, 1] = -1
        self.haar_weights[1, 1, 0, 1] = -1

        self.haar_weights[1, 0, 0, 2] = -1
        self.haar_weights[1, 1, 0, 2] = -1

        self.haar_weights[1, 0, 0, 3] = -1
        self.haar_weights[0, 1, 0, 3] = -1

        self.haar_weights = np.concatenate([self.haar_weights]*self.in_channels, axis=2)
        self.haar_weights = tf.constant(self.haar_weights)
        self.ident = tf.constant(tf.reshape(tf.eye(4 * self.in_channels),
                                    (1, 1, 4 * self.in_channels, 4 * self.in_channels)))

        self.permute = order_by_wavelet

        if self.permute:
            raise NotImplementedError("Ordering by wavelet not implemented for FrEIA Keras backend")

    def call(self, x, rev=False):
        x = x[0]
        if not rev:
            out = tf.nn.separable_conv2d(x, self.haar_weights, self.ident, strides=[1, 2, 2, 1], padding='VALID', data_format='NHWC')
            if self.permute:
                # TODO (unreachable)
                return [out[:, self.perm] * self.fac_fwd]
            else:
                return [out * self.fac_fwd]

        else:
            if self.permute:
                # TODO (unreachable)
                x_perm = x[:, self.perm_inv]
            else:
                x_perm = x

            # TODO wow...
            raise NotImplementedError(('wow.... grouped/separable transposed convolution'
                                       'simply does not exist in tensorflow :('))
            return None

    def jacobian(self, x, rev=False):
        return (-1)**int(rev) * self.jac

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        w, h, c = input_dims[0]
        c2, w2, h2 = c*4, w//2, h//2
        self.elements = c*w*h
        self.jac = self.elements / 4 * (np.log(16.) + 4 * np.log(self.fac_fwd))

        assert c*h*w == c2*h2*w2, "Uneven input width/height"
        return [(w2, h2, c2)]
