import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers

SEQUENCE_AXIS = -2


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


class ThresholdLayer(layers.Layer):
  def __init__(self, **kwargs):
    super(ThresholdLayer, self).__init__(**kwargs)
    self.kernel = None

  def build(self, input_shape):
    self.kernel = self.add_weight(name="threshold", shape=(1,), initializer="uniform",
                                  trainable=True)
    super(ThresholdLayer, self).build(input_shape)

  def call(self, x, *args, **kwargs):
    return keras.backend.sigmoid(100 * (x - self.kernel))

  def compute_output_shape(self, input_shape):
    return input_shape


class PositionEmbedding(layers.Layer):

  def __init__(
          self,
          sequence_length,
          initializer="glorot_uniform",
          **kwargs,
  ):
    super().__init__(**kwargs)
    if sequence_length is None:
      raise ValueError(
        "`sequence_length` must be an Integer, received `None`."
      )

    self.position_embeddings = None
    self.sequence_length = int(sequence_length)
    self.initializer = keras.initializers.get(initializer)

  def get_config(self):
    _config = super().get_config()
    _config.update(
      {
        "sequence_length": self.sequence_length,
        "initializer": keras.initializers.serialize(self.initializer),
      }
    )
    return _config

  def build(self, input_shape):
    feature_size = input_shape[-1]
    self.position_embeddings = self.add_weight(
      "embeddings",
      shape=[self.sequence_length, feature_size],
      initializer=self.initializer,
      trainable=True,
    )

    super().build(input_shape)

  def call(self, inputs, *args, **kwargs):
    if isinstance(inputs, tf.RaggedTensor):
      bounding_shape = inputs.bounding_shape()
      position_embeddings = self._trim_and_broadcast_position_embeddings(
        bounding_shape,
      )
      # then apply row lengths to recreate the same ragged shape as inputs
      return tf.RaggedTensor.from_tensor(
        position_embeddings,
        inputs.nested_row_lengths(),
      )
    else:
      return self._trim_and_broadcast_position_embeddings(
        tf.shape(inputs),
      )

  def _trim_and_broadcast_position_embeddings(self, shape):
    input_length = shape[SEQUENCE_AXIS]
    # trim to match the length of the input sequence, which might be less
    # than the sequence_length of the layer.
    position_embeddings = self.position_embeddings[:input_length, :]
    # then broadcast to add the missing dimensions to match "shape"
    return tf.broadcast_to(position_embeddings, shape)


class SinePositionEncoding(layers.Layer):

  def __init__(
          self,
          max_wavelength=10000,
          **kwargs,
  ):
    super().__init__(**kwargs)
    self.max_wavelength = max_wavelength

  def call(self, inputs, *args, **kwargs):
    # TODO(jbischof): replace `hidden_size` with`hidden_dim` for consistency
    # with other layers.
    input_shape = tf.shape(inputs)
    # length of sequence is the second last dimension of the inputs
    seq_length = input_shape[-2]
    hidden_size = input_shape[-1]
    position = tf.cast(tf.range(seq_length), self.compute_dtype)
    min_freq = tf.cast(1 / self.max_wavelength, dtype=self.compute_dtype)
    timescales = tf.pow(
      min_freq,
      tf.cast(2 * (tf.range(hidden_size) // 2), self.compute_dtype)
      / tf.cast(hidden_size, self.compute_dtype),
    )
    angles = tf.expand_dims(position, 1) * tf.expand_dims(timescales, 0)
    # even indices are sine, odd are cosine
    cos_mask = tf.cast(tf.range(hidden_size) % 2, self.compute_dtype)
    sin_mask = 1 - cos_mask
    # embedding shape is [seq_length, hidden_size]
    positional_encodings = (
            tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
    )

    return tf.broadcast_to(positional_encodings, input_shape)

  def get_config(self):
    _config = super().get_config()
    _config.update(
      {
        "max_wavelength": self.max_wavelength,
      }
    )
    return _config
