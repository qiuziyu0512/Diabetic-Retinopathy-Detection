"""
This file is the model used to train.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Define a ResNet Block
class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=2):
        super(BasicBlock, self).__init__()

        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv1 = layers.Conv2D(filter_num, kernel_size=[3, 3], strides=stride, padding='same', use_bias=False)

        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filter_num, kernel_size=[3, 3], strides=1, padding='same', use_bias=False)

        if stride != 1:
            self.skip_conv = layers.Conv2D(filter_num, kernel_size=[1, 1], strides=stride, use_bias=False)
            self.skip_bn = layers.BatchNormalization()

        self.stride = stride

    def call(self, inputs, training=True):

        out = self.bn1(inputs, training=training)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)

        if self.stride != 1:
            identity = self.skip_bn(inputs)
            identity = self.skip_conv(identity)
            out = out + identity
        else:
            out = out + inputs
        return out


# Define ResNet
class ResNet(keras.Model):

    def __init__(self, pre_channel_num=16):
        super(ResNet, self).__init__()

        self.layer1 = layers.Conv2D(pre_channel_num, (7, 7), strides=2, padding='same', use_bias=False)
        self.layer2 = layers.BatchNormalization()
        self.layer3 = layers.Activation('relu')
        self.layer4 = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')

        self.layer5 = BasicBlock(pre_channel_num)
        self.layer6 = BasicBlock(2 * pre_channel_num)
        self.layer7 = BasicBlock(4 * pre_channel_num)

        self.act = layers.Activation('relu')

        self.avgpool = layers.GlobalAveragePooling2D(data_format='channels_last')

        self.fc = layers.Dense(2, activation='softmax')

    def call(self, inputs, training=True):

        x = self.layer1(inputs)
        x = self.layer2(x, training=training)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer5(x, training=training)
        x = self.layer6(x, training=training)
        x = self.layer7(x, training=training)

        conv_output = self.act(x)

        x = self.avgpool(conv_output)

        x = self.fc(x)
        return x, conv_output


if __name__ == '__main__':
    model = ResNet()
    model.build(input_shape=(None, 256, 256, 3))
    model.summary()











