import warnings

import tensorflow.keras as kr
import tensorflow as tf

import numpy as np
from scipy.stats import special_ortho_group



class AllInOneBlock(kr.layers.Layer):
    ''' Combines affine coupling, permutation, global affine transformation ('ActNorm')
    in one block.'''

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor=None,
                 affine_clamping=2.,
                 gin_block=False,
                 global_affine_init=1.,
                 global_affine_type='SOFTPLUS',
                 permute_soft=False,
                 learned_householder_permutation=0,
                 reverse_permutation=False,
                 permutation_random_seed=None):
        '''
        subnet_constructor: class or callable f, called as
            f(channels_in, channels_out) and should return a tf.keras.layers.Layer

        affine_clamping: clamp the output of the mutliplicative coefficients
            (before exponentiation) to +/- affine_clamping.

        gin_block: Turn the block into a GIN block from Sorrenson et al, 2019

        global_affine_init: Initial value for the global affine scaling beta

        global_affine_init: 'SIGMOID', 'SOFTPLUS', or 'EXP'. Defines the activation
            to be used on the beta for the global affine scaling.

        permute_soft: bool, whether to sample the permutation matrices from SO(N),
            or to use hard permutations in stead. Note, permute_soft=True is very slow
            when working with >512 dimensions.

        reverse_permutation: Reverse the permutation before the block, as introduced by
            Putzky et al, 2019.

        permutation_random_seed: I not None: seed for the random permutation.
            Also applies to initialization of the permutation using householder reflections.
        '''

        super().__init__()

        channels = dims_in[0][-1]
        self.Ddim = len(dims_in[0]) - 1
        self.sum_dims = tuple(range(1, 2 + self.Ddim))

        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        elif len(dims_c) == 1:
            raise NotImplementedError("Keras cINN block doesn't work yet")
            #self.conditional = True
            #self.condition_channels = dims_c[0][0]
            #assert tuple(dims_c[0][1:]) == tuple(dims_in[0][1:]), \
                #F"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}."
        else:
            raise ValueError('Only supports one condition (concatenate externally)')

        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]

        assert learned_householder_permutation == 0, "doesn't work yet with Keras backend"

        self.in_channels = channels
        self.clamp = affine_clamping
        self.GIN = gin_block
        self.welling_perm = reverse_permutation

        self.permute_function = (lambda u, w: tf.linalg.matvec(w, u, transpose_a=True))

        if permute_soft and channels > 512:
            warnings.warn(("Soft permutation will take a very long time to initialize "
                           f"with {channels} feature channels. Consider using hard permutation instead."))

        if permute_soft:
            w = special_ortho_group.rvs(channels, random_state=permutation_random_seed).astype(np.float32)
        else:
            w = np.zeros((channels,channels)).astype(np.float32)
            np.random.seed(permutation_random_seed)
            for i,j in enumerate(np.random.permutation(channels)):
                w[i,j] = 1.
            np.random.seed()

        self.w = tf.Variable(w.reshape((*([1] * self.Ddim), channels, channels)),
                              trainable=False)
        self.w_inv = tf.Variable(w.T.reshape((*([1] * self.Ddim), channels, channels)),
                              trainable=False)

        if global_affine_type == 'SIGMOID':
            global_scale = np.log(global_affine_init)
            self.global_scale_activation = (lambda a: 10 * tf.sigmoid(a - 2.))
        elif global_affine_type == 'SOFTPLUS':
            global_scale = 10. * global_affine_init
            self.global_scale_activation = (lambda a: 0.1 * tf.nn.softplus(a))
        elif global_affine_type == 'EXP':
            global_scale = np.log(global_affine_init)
            self.global_scale_activation = (lambda a: tf.exp(a))
        else:
            raise ValueError('Please, SIGMOID, SOFTPLUS or EXP, as global affine type')

        self.global_scale = tf.Variable(tf.ones((1, *([1] * self.Ddim), self.in_channels)) * float(global_scale))
        self.global_offset = tf.Variable(tf.zeros((1,  *([1] * self.Ddim), self.in_channels)))

        self.s = subnet_constructor(self.splits[0] + self.condition_channels, 2 * self.splits[1])
        self.last_jac = None

    def build(self, input_shape):
        # TODO oof... flaky... where does this come from?
        input_shape = input_shape[0]

        self.s.build((None, input_shape[0], input_shape[1], self.splits[0] + self.condition_channels))
        super().build(input_shape)

    def log_e(self, s):
        s = self.clamp * tf.math.tanh(0.1 * s)
        if self.GIN:
            s -= tf.math.reduce_mean(s, axis=self.sum_dims, keepdims=True)
        return s

    def permute(self, x, rev=False):
        if self.GIN:
            scale = 1.
        else:
            scale = self.global_scale_activation(self.global_scale)
        if rev:
            return (self.permute_function(x, self.w_inv) - self.global_offset) / scale
        else:
            return self.permute_function(x * scale + self.global_offset, self.w)

    def pre_permute(self, x, rev=False):
        if rev:
            return self.permute_function(x, self.w)
        else:
            return self.permute_function(x, self.w_inv)

    def affine(self, x, a, rev=False):
        ch = x.shape[-1]
        s, t = tf.split(a, (ch, ch), axis=-1)
        sub_jac = self.log_e(s)
        if not rev:
            return (x * tf.math.exp(sub_jac) + 0.1 * t,
                    tf.math.reduce_sum(sub_jac, axis=self.sum_dims))
        else:
            return ((x - 0.1 * t) * tf.math.exp(-sub_jac),
                    -tf.math.reduce_sum(sub_jac, axis=self.sum_dims))

    def call(self, x, c=[], rev=False):

        if rev:
            x = [self.permute(x[0], rev=True)]
        elif self.welling_perm:
            x = [self.pre_permute(x[0], rev=False)]

        x1, x2 = tf.split(x[0], self.splits, axis=-1)

        if self.conditional:
            x1c = tf.concat([x1, *c], -1)
        else:
            x1c = x1

        if not rev:
            a1 = self.s(x1c)
            x2, j2 = self.affine(x2, a1)
        else:
            # names of x and y are swapped!
            a1 = self.s(x1c)
            x2, j2 = self.affine(x2, a1, rev=True)

        self.last_jac = j2
        x_out = tf.concat((x1, x2), -1)

        n_pixels = 1
        for d in self.sum_dims[:-1]:
            n_pixels *= x_out.shape[d]
        self.last_jac += ((-1)**rev * n_pixels) * (1 - int(self.GIN)) * tf.reduce_sum(tf.math.log(self.global_scale_activation(self.global_scale) + 1e-12))

        if not rev:
            x_out = self.permute(x_out, rev=False)
        elif self.welling_perm:
            x_out = self.pre_permute(x_out, rev=True)

        return [x_out]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims
