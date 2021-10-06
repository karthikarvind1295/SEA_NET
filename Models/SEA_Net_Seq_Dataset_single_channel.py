"""This program uses a SEA Net trained  with all the pick locations for each demonstrations"""

"""
SEquential Attention Network Module - Sequential Dataset - ONE CHANNEL TRAINED
"""

import numpy as np

from Models_tf.resnet_with_regression import ResNet43_8s_classification_regression
import tensorflow as tf
from utils import preprocess
import cv2


class SEA_Net_Seq_Dataset:
  """
    SEquential Attention Network Module - Sequential Dataset
  """

  def __init__(self, in_shape, num_labels, lite=False):
    """
    The class accepts initialises the Sequential Attention Network
    :param in_shape: The input shape of the image
    :param num_labels: The number of objects for sequential picks
    """

    max_dim = np.max(in_shape[:2])

    self.padding = np.zeros((3, 2), dtype=int)
    pad = (max_dim - np.array(in_shape[:2])) / 2
    self.padding[:2] = pad.reshape(2, 1)
    
    self.num_labels = num_labels

    in_shape = np.array(in_shape)
    in_shape += np.sum(self.padding, axis=1)
    in_shape = tuple(in_shape)
    

    # Initialize fully convolutional Residual Network with 43 layers and
    # 8-stride (3 2x2 max pools and 3 2x bilinear upsampling)

    d_in, d_out, d_reg = ResNet43_8s_classification_regression(in_shape, self.num_labels, prefix='Attention_')

    self.model = tf.keras.models.Model(inputs=[d_in], outputs=[d_out, d_reg])
    self.optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
    self.metric = tf.keras.metrics.Mean(name='loss_attention')
    self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0, reduction=tf.keras.losses.Reduction.NONE)
    #tf.keras.utils.plot_model(self.model, to_file='model.png', show_shapes=False, show_dtype=False, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)


  def forward(self, in_img, softmax=True):
    """Forward pass."""
    # Forward pass
    in_data = np.pad(in_img, self.padding, mode='constant')
    in_data = preprocess(in_data)
    in_shape = (1,) + in_data.shape
    in_data = in_data.reshape(in_shape)
    in_tens = tf.convert_to_tensor(in_data, dtype=tf.float32)

    
    logits, logits_classification = self.model(in_tens)


    batch, height, width, channels = logits.shape
    output = tf.reshape(logits, (height, width, channels))
    if softmax:
        """
        Take a softmax for each channel in the output logits
        """
        for i in range(channels):
          first_channel = np.reshape(output[:, :, i], (1, np.prod(output[:, :, i].shape)))
          first_channel = tf.nn.softmax(first_channel)
          first_channel = np.reshape(first_channel, (224, 224,1))
          output[:, :, i] = first_channel
        
        logits_classification = tf.nn.softmax(logits_classification)

    
    return output, logits, logits_classification

  def train(self, in_img, goal_location, goal_label, real_time=False):
    """Train."""
    loss_1 = 0.0
    w = 0.6
    self.metric.reset_states()
    with tf.GradientTape() as tape:

      num_labels = self.num_labels  
      output, _, logits_classification = self.forward(in_img, softmax=False)

      #Get labels
      label_size = in_img.shape[:2] + (num_labels,)
      label = np.zeros(label_size)
      label_clasif = np.zeros((1, self.num_labels))
      #print(logits_classification.shape)
      #initial_channel = 0

      for ij in range(0, num_labels):
        if real_time:
          label[int(goal_location[ij][0][1]), int(goal_location[ij][0][0]), int(ij)] = 1 #For Real time classification
        else:
          label[int(goal_location[ij][0]), int(goal_location[ij][1]), int(ij)] = 1 
        #label[int(goal_location[ij][0]), int(goal_location[ij][1]), int(ij)] = 1
      label_clasif[:, int(goal_label)] = 1

      label = tf.convert_to_tensor(label, dtype=tf.float32)
      label_clasif = tf.convert_to_tensor(label_clasif, dtype=tf.float32)
      tf.stop_gradient(label)
      tf.stop_gradient(label_clasif)
      # Get loss.
      
      for k in range(num_labels):
          if k == goal_label: #Only the loss corresponding to the goal_label is trained
            output_temp = output[:, :, k]
            output_temp = tf.reshape(output_temp, (1, np.prod(output_temp.shape)))

            label_temp = label[:, :, k]
            label_temp = tf.reshape(label_temp, (1, np.prod(label_temp.shape)))
            loss_1 = tf.nn.softmax_cross_entropy_with_logits(labels = label_temp, logits = output_temp)
      
      loss_2 = self.loss_fn(label_clasif, logits_classification)

      #loss = w * loss_1 + (1 - w) * loss_2
      loss = loss_1 + loss_2
      loss = tf.reduce_mean(loss)

    # Backpropagate
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))

    self.metric(loss)

    return np.float32(loss)

  def load(self, path):
      self.model.load_weights(path)

  def save(self, filename):
      self.model.save(filename)

  def get_attention_heatmap(self, attention):
    """Given attention, get a human-readable heatmap.
    https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html  
    In my normal usage, the attention is already softmax-ed but just be
    aware in case it's not. Also be aware of RGB vs BGR mode. We should
    ensure we're in BGR mode before saving. Also with RAINBOW mode, red =
    hottest (highest attention values), green=medium, blue=lowest.
    Note: to see the grayscale only (which may be easier to interpret,
    actually...) save `vis_attention` just before applying the colormap.
    """
    # Options: cv2.COLORMAP_PLASMA, cv2.COLORMAP_JET, etc.
    #attention = tf.reshape(attention, (1, np.prod(attention.shape)))
    #attention = tf.nn.softmax(attention)
    vis_attention = np.float32(attention).reshape((320, 160))
    vis_attention = vis_attention - np.min(vis_attention)
    vis_attention = 255 * vis_attention / np.max(vis_attention)
    vis_attention = cv2.applyColorMap(np.uint8(vis_attention), cv2.COLORMAP_RAINBOW)
    vis_attention = cv2.cvtColor(vis_attention, cv2.COLOR_RGB2BGR)
    return vis_attention

