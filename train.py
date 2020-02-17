"""
This file is the training routine
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import gin.tf
import datetime
from data import make_dataset
from model import ResNet
from data_augmentation import data_augmentation
from absl import logging
from metrics import ConfusionMatrix
from time import time
import matplotlib.pyplot as plt
import numpy as np
import itertools
import io
import tensorflow as tf


logging.set_verbosity(logging.INFO)


# Define loss function
def loss(model, x, y):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    y_pred, _ = model(x)
    return loss_object(y_true=y, y_pred=y_pred)


# Define gradient
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def plot_confusion_matrix(cm, class_names):
    """
    The function is to return a  matplotlib figure containing the plotted confusion matrix.

    Args:
        cm(array, shape = [n, n]): a confusion matrix of integer classes
        class_names(array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(7, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix
    cm = cm.numpy()
    cm = np.around(cm.astype('float')/cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black
    threshold = cm.max() / 1.3
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = 'white' if cm[i,j] > threshold else 'black'
        plt.text(j, i, cm[i, j], horizontalalignment='center', color=color)

    plt.ylim([1.5, -0.5])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return figure


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and returns it.
    The supplied figure is closed and inaccessible after the call
    """
    # Save the plot to a PNG in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    return image


def plot_map(img, act_map, pred_prob, pred_label, true_label, name):
    if pred_label:
        pred_label = 'RDR'
    else:
        pred_label = 'NRDR'
    if true_label:
        true_label = 'RDR'
    else:
        true_label = 'NRDR'
    img = tf.squeeze(img, axis=0)
    act_map = tf.squeeze(act_map, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(img)
    axes[1].imshow(img)
    i = axes[1].imshow(act_map, cmap="jet", alpha=0.5)
    fig.colorbar(i)
    plt.suptitle("File {} Pr(class={})= {:5.2f} Ground truth {}".format(
        name.numpy().decode('utf-8'), pred_label, pred_prob[0], true_label))
    plt.savefig('{}_detection.png'.format(name.numpy().decode('utf-8')))
    plt.show()


# Define training function
def train(hparams, num_epoch, tuning):

    log_dir = './results/'
    test_batch_size = 8
    # Load dataset
    training_set, valid_set = make_dataset(BATCH_SIZE=hparams['HP_BS'], file_name='train_tf_record', split=True)
    test_set = make_dataset(BATCH_SIZE=test_batch_size, file_name='test_tf_record', split=False)
    class_names = ['NRDR', 'RDR']

    # Model
    model = ResNet()

    # set optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['HP_LR'])
    # set metrics
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    valid_accuracy = tf.keras.metrics.Accuracy()
    valid_con_mat = ConfusionMatrix(num_class=2)
    test_accuracy = tf.keras.metrics.Accuracy()
    test_con_mat = ConfusionMatrix(num_class=2)

    # Save Checkpoint
    if not tuning:
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
        manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=5)

    # Set up summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log_dir = log_dir + current_time + '/train'
    summary_writer = tf.summary.create_file_writer(tb_log_dir)

    # Restore Checkpoint
    if not tuning:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            logging.info('Restored from {}'.format(manager.latest_checkpoint))
        else:
            logging.info('Initializing from scratch.')

    @tf.function
    def train_step(train_img, train_label):
        # Optimize the model
        loss_value, grads = grad(model, train_img, train_label)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_pred, _ = model(train_img)
        train_label = tf.expand_dims(train_label, axis=1)
        train_accuracy.update_state(train_label, train_pred)

    for epoch in range(num_epoch):

        begin = time()

        # Training loop
        for train_img, train_label, train_name in training_set:
            train_img = data_augmentation(train_img)
            train_step(train_img, train_label)

        with summary_writer.as_default():
            tf.summary.scalar('Train Accuracy', train_accuracy.result(), step=epoch)

        for valid_img, valid_label, _ in valid_set:
            valid_img = tf.cast(valid_img, tf.float32)
            valid_img = valid_img / 255.0
            valid_pred, _ = model(valid_img, training=False)
            valid_pred = tf.cast(tf.argmax(valid_pred, axis=1), dtype=tf.int64)
            valid_con_mat.update_state(valid_label, valid_pred)
            valid_accuracy.update_state(valid_label, valid_pred)

        # Log the confusion matrix as an image summary
        cm_valid = valid_con_mat.result()
        figure = plot_confusion_matrix(cm_valid, class_names=class_names)
        cm_valid_image = plot_to_image(figure)

        with summary_writer.as_default():
            tf.summary.scalar('Valid Accuracy', valid_accuracy.result(), step=epoch)
            tf.summary.image('Valid ConfusionMatrix', cm_valid_image, step=epoch)

        end = time()
        logging.info("Epoch {:d} Training Accuracy: {:.3%} Validation Accuracy: {:.3%} Time:{:.5}s".format(epoch+1,
                     train_accuracy.result(), valid_accuracy.result(), (end - begin)))
        train_accuracy.reset_states()
        valid_accuracy.reset_states()
        valid_con_mat.reset_states()
        if not tuning:
            if int(ckpt.step) % 5 == 0:
                save_path = manager.save()
                logging.info('Saved checkpoint for epoch {}: {}'.format(int(ckpt.step), save_path))
            ckpt.step.assign_add(1)

    for test_img, test_label, _ in test_set:
        test_img = tf.cast(test_img, tf.float32)
        test_img = test_img / 255.0
        test_pred, _ = model(test_img, training=False)
        test_pred = tf.cast(tf.argmax(test_pred, axis=1), dtype=tf.int64)
        test_accuracy.update_state(test_label, test_pred)
        test_con_mat.update_state(test_label, test_pred)

    cm_test = test_con_mat.result()
    # Log the confusion matrix as an image summary
    figure = plot_confusion_matrix(cm_test, class_names=class_names)
    cm_test_image = plot_to_image(figure)
    with summary_writer.as_default():
        tf.summary.scalar('Test Accuracy', test_accuracy.result(), step=epoch)
        tf.summary.image('Test ConfusionMatrix', cm_test_image, step=epoch)

    logging.info("Trained finished. Final Accuracy in test set: {:.3%}".format(test_accuracy.result()))

    # Visualization
    if not tuning:
        for vis_img, vis_label, vis_name in test_set:
            vis_label = vis_label[0]
            vis_name = vis_name[0]
            vis_img = tf.cast(vis_img[0], tf.float32)
            vis_img = tf.expand_dims(vis_img, axis=0)
            vis_img = vis_img / 255.0
            with tf.GradientTape() as tape:
                vis_pred, conv_output = model(vis_img, training=False)
                pred_label = tf.argmax(vis_pred, axis=-1)
                vis_pred = tf.reduce_max(vis_pred, axis=-1)
                grad_1 = tape.gradient(vis_pred, conv_output)
                weight = tf.reduce_mean(grad_1, axis=[1, 2]) / grad_1.shape[1]
                act_map0 = tf.nn.relu(tf.reduce_sum(weight * conv_output, axis=-1))
                act_map0 = tf.squeeze(tf.image.resize(tf.expand_dims(act_map0, axis=-1), (256, 256), antialias=True),
                                      axis=-1)
                plot_map(vis_img, act_map0, vis_pred, pred_label, vis_label, vis_name)
            break

    return test_accuracy.result()



















