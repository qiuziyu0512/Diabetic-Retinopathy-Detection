"""
This file is the main function
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import gin.tf
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from train import train


def run(run_dir, hparams, epoch, tuning):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train(hparams, epoch, tuning)
        tf.summary.scalar('accuracy', accuracy, step=10)


@gin.configurable
def main(hparams, num_epoch, tuning):
    if tuning:
        # Hyperparameter tuning, multiple run
        HP_BS = hp.HParam('train_batch_size', hp.Discrete([2, 4]))
        HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-5, 1e-4, 1e-3, 1e-2]))

        with tf.summary.create_file_writer('results/hparam_tuning').as_default():
            hp.hparams_config(hparams=[HP_BS, HP_LR], metrics=[hp.Metric('accuracy', display_name='Accuracy')])

        session_num = 0
        for batch_size in HP_BS.domain.values:
            for learning_rate in HP_LR.domain.values:
                hparams = {'HP_BS': batch_size,
                           'HP_LR': learning_rate}
                run_name = "run-{:d}".format(session_num)
                print('--- Starting trial: %s' % run_name)
                print('Batch_size = {}, learning_rate = {}'.format(batch_size, learning_rate))
                run('results/hparam_tuning/' + run_name, hparams, num_epoch, tuning)
                session_num += 1
    else:
        train(hparams, num_epoch, tuning)


if __name__ == "__main__":
    gpu = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    gin.parse_config_file('config.gin')
    main()



