import os

import tensorflow as tf
import tensorflow.keras as kr
import numpy as np
#from tensorflow.compat.v1 import disable_eag

from model import build_model
from data import Dataset


def train(args):

    use_tarantella   = eval(args['training']['use_tarantella'])
    n_GPUs           = eval(args['training']['tarantella_GPUs'])
    ndims_tot        = np.prod(eval(args['data']['data_dimensions']))
    output_dir       = args['checkpoints']['output_dir']
    sched_milestones = eval(args['training']['milestones_lr_decay'])
    n_epochs         = eval(args['training']['N_epochs'])
    optimizer_kwargs = eval(args['training']['optimizer_kwargs'])
    optimizer_type   = args['training']['optimizer']
    optimizer_lr     = eval(args['training']['lr'])

    if use_tarantella:
        import tarantella
        tarantella.init(n_GPUs)
        node_rank = tarantella.get_rank()
        nodes_number = tarantella.get_size()
    else:
        node_rank = 0
        nodes_number = 1
    is_primary_node = (node_rank == 0)

    model = build_model(args)
    data = Dataset(args)

    def nll_loss_z_part(y, z):
        zz = tf.math.reduce_mean(z**2)
        return 0.5 * zz

    def nll_loss_jac_part(y, jac):
        return - tf.math.reduce_mean(jac) / ndims_tot

    def lr_sched(ep, lr):
        if ep in sched_milestones:
            return 0.1 * lr
        return lr

    # TODO: should this only be for one node, or for each?
    lr_scheduler_callback = kr.callbacks.LearningRateScheduler(lr_sched, verbose=is_primary_node)


    callbacks = [lr_scheduler_callback, kr.callbacks.TerminateOnNaN()]

    if is_primary_node:
        checkpoint_callback = kr.callbacks.ModelCheckpoint(filepath=os.path.join(output_dir, 'checkpoint_best.hdf5'),
                                                           save_best_only=True,
                                                           save_weights_only=True,
                                                           mode='min',
                                                           verbose=is_primary_node)
        loss_log_callback = kr.callbacks.CSVLogger(os.path.join(output_dir, 'losses.dat'), separator=' ')

        callbacks.append(checkpoint_callback)
        callbacks.append(loss_log_callback)


    try:
        optimizer_type = {'ADAM': kr.optimizers.Adam,
                          'SGD': kr.optimizers.SGD
                         }[optimizer_type]
    except KeyError:
        optimizer_type = eval(optimizer_type)

    optimizer = optimizer_type(optimizer_lr, **optimizer_kwargs)

    model.compile(loss=[nll_loss_z_part, nll_loss_jac_part],
                  optimizer=optimizer)

    history = model.fit(data.train_dataset,
                        epochs  = n_epochs,
                        verbose = is_primary_node,
                        callbacks = [lr_scheduler_callback, checkpoint_callback],
                        validation_data = data.test_dataset)

    model.save_weights(os.path.join(output_dir, 'checkpoint_end.hdf5'), overwrite=True)
