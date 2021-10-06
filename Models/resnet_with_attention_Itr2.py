# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resnet module."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import tensorflow as tf
import numpy as np


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   activation=True,
                   include_batchnorm=False):
  """The identity block is the block that has no conv layer at shortcut.

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of
        middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    activation: If True, include ReLU activation on the output.
    include_batchnorm: If True, include intermediate batchnorm layers.

  Returns:
    Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  batchnorm_axis = 3
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = tf.keras.layers.Conv2D(
      filters1, (1, 1),
      dilation_rate=(1, 1),
      kernel_initializer='glorot_uniform',
      name=conv_name_base + '2a')(
          input_tensor)
  if include_batchnorm:
    x = tf.keras.layers.BatchNormalization(
        axis=batchnorm_axis, name=bn_name_base + '2a')(
            x)
  x = tf.keras.layers.ReLU()(x)

  x = tf.keras.layers.Conv2D(
      filters2,
      kernel_size,
      dilation_rate=(1, 1),
      padding='same',
      kernel_initializer='glorot_uniform',
      name=conv_name_base + '2b')(
          x)
  if include_batchnorm:
    x = tf.keras.layers.BatchNormalization(
        axis=batchnorm_axis, name=bn_name_base + '2b')(
            x)
  x = tf.keras.layers.ReLU()(x)

  x = tf.keras.layers.Conv2D(
      filters3, (1, 1),
      dilation_rate=(1, 1),
      kernel_initializer='glorot_uniform',
      name=conv_name_base + '2c')(
          x)
  if include_batchnorm:
    x = tf.keras.layers.BatchNormalization(
        axis=batchnorm_axis, name=bn_name_base + '2c')(
            x)

  x = tf.keras.layers.add([x, input_tensor])

  if activation:
    x = tf.keras.layers.ReLU()(x)
  return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               activation=True,
               include_batchnorm=False):
  """A block that has a conv layer at shortcut.

  Note that from stage 3,
  the first conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of
        middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the first conv layer in the block.
    activation: If True, include ReLU activation on the output.
    include_batchnorm: If True, include intermediate batchnorm layers.

  Returns:
    Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  batchnorm_axis = 3
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = tf.keras.layers.Conv2D(
      filters1, (1, 1),
      strides=strides,
      dilation_rate=(1, 1),
      kernel_initializer='glorot_uniform',
      name=conv_name_base + '2a')(
          input_tensor)
  if include_batchnorm:
    x = tf.keras.layers.BatchNormalization(
        axis=batchnorm_axis, name=bn_name_base + '2a')(
            x)
  x = tf.keras.layers.ReLU()(x)

  x = tf.keras.layers.Conv2D(
      filters2,
      kernel_size,
      padding='same',
      dilation_rate=(1, 1),
      kernel_initializer='glorot_uniform',
      name=conv_name_base + '2b')(
          x)
  if include_batchnorm:
    x = tf.keras.layers.BatchNormalization(
        axis=batchnorm_axis, name=bn_name_base + '2b')(
            x)
  x = tf.keras.layers.ReLU()(x)

  x = tf.keras.layers.Conv2D(
      filters3, (1, 1),
      kernel_initializer='glorot_uniform',
      dilation_rate=(1, 1),
      name=conv_name_base + '2c')(
          x)
  if include_batchnorm:
    x = tf.keras.layers.BatchNormalization(
        axis=batchnorm_axis, name=bn_name_base + '2c')(
            x)

  shortcut = tf.keras.layers.Conv2D(
      filters3, (1, 1),
      strides=strides,
      dilation_rate=(1, 1),
      kernel_initializer='glorot_uniform',
      name=conv_name_base + '1')(
          input_tensor)
  if include_batchnorm:
    shortcut = tf.keras.layers.BatchNormalization(
        axis=batchnorm_axis, name=bn_name_base + '1')(
            shortcut)

  x = tf.keras.layers.add([x, shortcut])
  if activation:
    x = tf.keras.layers.ReLU()(x)
  return x

def hw_flatten(x) :
    return tf.reshape(x, shape=[1, x.shape[1] * x.shape[2], x.shape[-1]]) #Assumed 1 as the batch_size

def attention(x, channels, output_dim, name='attention'):
    # with tf.variable_scope(scope):
    f = tf.keras.layers.Conv2D(channels  , kernel_size=1, strides=1, kernel_initializer='glorot_uniform', name='f_conv')(x) # [bs, h, w, c'] 
    g = tf.keras.layers.Conv2D(channels , kernel_size=1, strides=1, kernel_initializer='glorot_uniform', name='g_conv')(x) # [bs, h, w, c']
    h = tf.keras.layers.Conv2D(channels , kernel_size=1, strides=1, kernel_initializer='glorot_uniform', name='h_conv')(x)
    #h = conv(x, channels, kernel=1, stride=1, sn=self.sn, scope='h_conv') # [bs, h, w, c]
    print("G shape {}".format(g.shape))
    
    # N = h * w
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

    beta = tf.nn.softmax(s)  # attention map

    o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
    #gamma = tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
    

    o = tf.reshape(o, shape=[1, x.shape[1], x.shape[2], x.shape[3]]) # [bs, h, w, C]
    o = tf.keras.layers.Conv2D(output_dim , kernel_size=1, strides=1, kernel_initializer='glorot_uniform', name='attn_conv')(o)
    #o = conv(o, channels, kernel=1, stride=1, sn=self.sn, scope='attn_conv')
    print("o shape {}".format(o.shape))
    #x = gamma * o + x

    return o


def ResNet43_8s_Attn(input_shape,  # pylint: disable=invalid-name
                output_dim,
                num_shapes,
                include_batchnorm=False,
                batchnorm_axis=3,
                prefix='',
                cutoff_early=False):
  """Build Resent 43 8s."""
  # TODO(andyzeng): rename to ResNet36_4s

  input_data = tf.keras.layers.Input(shape=input_shape)

  x = tf.keras.layers.Conv2D(
      64, (3, 3),
      strides=(1, 1),
      padding='same',
      kernel_initializer='glorot_uniform',
      name=prefix + 'conv1')(
          input_data)
  if include_batchnorm:
    x = tf.keras.layers.BatchNormalization(
        axis=batchnorm_axis, name=prefix + 'bn_conv1')(
            x)
  x = tf.keras.layers.ReLU()(x)

  if cutoff_early:
    x = conv_block(
        x,
        5, [64, 64, output_dim],
        stage=2,
        block=prefix + 'a',
        strides=(1, 1),
        include_batchnorm=include_batchnorm)
    x = identity_block(
        x,
        5, [64, 64, output_dim],
        stage=2,
        block=prefix + 'b',
        include_batchnorm=include_batchnorm)
    return input_data, x

  x = conv_block(
      x, 3, [64, 64, 64], stage=2, block=prefix + 'a', strides=(1, 1))
  x = identity_block(x, 3, [64, 64, 64], stage=2, block=prefix + 'b')

  x = conv_block(
      x, 3, [128, 128, 128], stage=3, block=prefix + 'a', strides=(2, 2))
  x = identity_block(x, 3, [128, 128, 128], stage=3, block=prefix + 'b')

  x = conv_block(
      x, 3, [256, 256, 256], stage=4, block=prefix + 'a', strides=(2, 2))
  x = identity_block(x, 3, [256, 256, 256], stage=4, block=prefix + 'b')

  x = conv_block(
      x, 3, [512, 512, 512], stage=5, block=prefix + 'a', strides=(2, 2))
  x = identity_block(x, 3, [512, 512, 512], stage=5, block=prefix + 'b')

  x = conv_block(
      x, 3, [256, 256, 256], stage=6, block=prefix + 'a', strides=(1, 1))
  x = identity_block(x, 3, [256, 256, 256], stage=6, block=prefix + 'b')

  x = tf.keras.layers.UpSampling2D(
      size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_1')(
          x)

  x = conv_block(
      x, 3, [128, 128, 128], stage=7, block=prefix + 'a', strides=(1, 1))
  x = identity_block(x, 3, [128, 128, 128], stage=7, block=prefix + 'b')

  x = tf.keras.layers.UpSampling2D(
      size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_2')(
          x)

  x = conv_block(
      x, 3, [64, 64, 64], stage=8, block=prefix + 'a', strides=(1, 1))
  x = identity_block(x, 3, [64, 64, 64], stage=8, block=prefix + 'b')

  x = tf.keras.layers.UpSampling2D(
      size=(2, 2), interpolation='bilinear', name=prefix + 'upsample_3')(
          x)

  x = conv_block(
      x,
      3, [16, 16, num_shapes],
      stage=9,
      block=prefix + 'a',
      strides=(1, 1),
      activation=False)
  x = identity_block(
      x, 3, [16, 16, num_shapes], stage=9, block=prefix + 'b', activation=False)

  output = attention(x, num_shapes, output_dim) #Attention in ResNet

  return input_data, output

