import keras.backend as K
import tensorflow as tf
from keras.engine import Layer


class Mag(Layer):
  """
  computes magnitudes of channels vectors. Output shape is (b, x, e) --> (b, x, 1)
  """

  def __init__(self, **kwargs):
    super(Mag, self).__init__(**kwargs)
    self.supports_masking = False

  def call(self, inputs, **kwargs):
    _sum = K.sum(K.square(inputs), axis=-1)
    return K.expand_dims(K.sqrt(_sum), -1)

  def compute_output_shape(self, input_shape):
    sh = list(input_shape)[:-1] + [1]
    print('input_shape', input_shape)
    print('sh', sh)
    return tuple(sh)


def sigmoid_focal_crossentropy(
        y_true,
        y_pred,
        alpha=0.25,
        gamma=2.0,
        from_logits=False,
):
  """
  Args
      y_true: true targets tensor.
      y_pred: predictions tensor.
      alpha: balancing factor.
      gamma: modulating factor.
  Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
      same shape as `y_true`; otherwise, it is scalar.
  """
  if gamma and gamma < 0:
    raise ValueError("Value of gamma should be greater than or equal to zero")

  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)

  # Get the cross_entropy for each entry
  ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

  # If logits are provided then convert the predictions into probabilities
  if from_logits:
    pred_prob = tf.sigmoid(y_pred)
  else:
    pred_prob = y_pred

  p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
  alpha_factor = 1.0
  modulating_factor = 1.0

  if alpha:
    alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

  if gamma:
    gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
    modulating_factor = tf.pow((1.0 - p_t), gamma)

  # compute the final loss and return
  return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)
