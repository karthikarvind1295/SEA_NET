
"""Adding self attention module for pick location"""

import numpy as np
from Models_tf import ResNet36_4s
from .resnet_with_attention_Itr1 import ResNet43_8s_Attn
#from ravens.utils import utils
import tensorflow as tf
from tensorflow_addons import image as tfa_image
from utils import preprocess
import cv2


class Multi_output_Self_Attention:
  """Attention module."""

  def __init__(self, in_shape, num_labels, lite=False):
    """
    The module accepts the shape of the input image and computes the attention module.
    :param in_shape: Shape of the input image
    :param num_labels: Number of output channel
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
    # 8-stride (3 2x2 max pools and 3 2x bilinear upsampling) with self-attention
    if lite:
        d_in, d_out = ResNet43_8s_Attn(in_shape, self.num_labels)
    else:
        d_in, d_out = ResNet43_8s_Attn(in_shape, self.num_labels)

    self.model = tf.keras.models.Model(inputs=[d_in], outputs=[d_out])
    self.optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
    self.metric = tf.keras.metrics.Mean(name='loss_attention')


  def forward(self, in_img, softmax=True):
    """Forward pass."""
    # Forward pass
    in_data = np.pad(in_img, self.padding, mode='constant')
    in_data = preprocess(in_data)
    in_shape = (1,) + in_data.shape
    in_data = in_data.reshape(in_shape)
    in_tens = tf.convert_to_tensor(in_data, dtype=tf.float32)

    
    logits = self.model(in_tens)

    batch, height, width, channels = logits.shape
    output = tf.reshape(logits, (height, width, channels))
    if softmax:
        output = tf.nn.softmax(output)
        #print(output.shape)
        #output = np.float32(output).reshape(np.product(1,logits.shape))
    
    return output, logits

  def train(self, in_img, goal_location):
    """Train."""
    loss = 0.0
    self.metric.reset_states()
    with tf.GradientTape() as tape:

      num_labels = self.num_labels 
      output, _ = self.forward(in_img, softmax=False)

      #Get labels
      label_size = in_img.shape[:2] + (num_labels,)
      label = np.zeros(label_size)
      #label[int(goal_label[0,1]), int(goal_label[0,0])] = 1

      # Convert to one-hot tensors
      for ij in range(0, num_labels):
        label[int(goal_location[ij][0]), int(goal_location[ij][1]), int(ij)] = 1
    
      label = tf.convert_to_tensor(label, dtype=tf.float32)
      tf.stop_gradient(label)

      # Get loss.
      for k in range(num_labels):
        output_temp = output[:, :, k]
        output_temp = tf.reshape(output_temp, (1, np.prod(output_temp.shape)))

        label_temp = label[:, :, k]
        label_temp = tf.reshape(label_temp, (1, np.prod(label_temp.shape)))
        loss += tf.nn.softmax_cross_entropy_with_logits(labels = label_temp, logits = output_temp)
      
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

