import torch.nn as nn
import torch

import tensorflow.keras as kr
import tensorflow as tf


class ReversibleSequential(kr.models.Model):

    def __init__(self, *dims):
        super().__init__()

        self.shapes = [tuple(dims)]
        self.conditions = []
        self.blocks_list = []

    def append(self, block_class, cond=None, cond_shape=None, **kwargs):

        dims_in = [self.shapes[-1]]
        assert cond is None, "TODO conditioning"

        self.conditions.append(cond)

        if cond is not None:
            kwargs['dims_c'] = [cond_shape]

        module = block_class(dims_in, **kwargs)
        self.blocks_list.append(module)
        ouput_dims = module.output_dims(dims_in)

        assert len(ouput_dims) == 1, "Module has more than one output"
        self.shapes.append(ouput_dims[0])


    def call(self, x, c=None, rev=False, intermediate_outputs=False):

        iterator = range(len(self.blocks_list))
        jac = None

        if rev:
            iterator = reversed(iterator)

        for i in iterator:
            if self.conditions[i] is None:
                x, j = (self.blocks_list[i]([x], rev=rev)[0],
                        self.blocks_list[i].jacobian([x], rev=rev))
            else:
                x, j = (self.blocks_list[i]([x], c=[c[self.conditions[i]]], rev=rev)[0],
                        self.blocks_list[i].jacobian([x], c=[c[self.conditions[i]]], rev=rev))

            # i would have liked to have done it with zeros....
            # but tensorflow, it gives me trouble...
            if jac is None:
                jac = j
            else:
                jac += j

        return x, jac
