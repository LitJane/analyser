from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf
from pandas import DataFrame
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import LayerNormalization, Input, Conv1D, Dropout, LSTM, Bidirectional, Dense, \
  MaxPooling1D, ReLU, LeakyReLU
from tensorflow.keras.layers import concatenate

from analyser.headers_detector import TOKEN_FEATURES
from analyser.hyperparams import work_dir
from analyser.structures import ContractSubject
from tf_support.addons import sigmoid_focal_crossentropy
from tf_support.tools import KerasTrainingContext

seq_labels_dn = ['date', 'number']
seq_labels_org_1 = ['org-1-name', 'org-1-type', 'org-1-alias']
seq_labels_org_2 = ['org-2-name', 'org-2-type', 'org-2-alias']
seq_labels_val = ['sign_value_currency/value', 'sign_value_currency/currency', 'sign_value_currency/sign']

seq_labels_contract_level_1 = [
  'headline_h1',
  'subject',
  '_reserved'
]

losses = {
  "O1_tagging": "binary_crossentropy",
  "O2_subject": "binary_crossentropy",
}

metrics = ['mse', 'binary_crossentropy']

t_semantic_map_keys_common = [
  'headline',
  'subject',
  'date',
  'number',
]
t_semantic_map_keys_org = [
  'org-name',
  'org-alias',
  'org-type']

t_semantic_map_keys_price = [
  'amount',
  'amount_brutto',
  'amount_netto',
  'vat',
  'sign',
  'currency',
  'vat_unit']

semantic_map_keys = t_semantic_map_keys_common + t_semantic_map_keys_org + t_semantic_map_keys_price + ['value']
semantic_map_keys_contract = []
for _name in semantic_map_keys:
  semantic_map_keys_contract.append(_name + "-begin")
  semantic_map_keys_contract.append(_name + "-end")

DEFAULT_TRAIN_CTX = KerasTrainingContext()

CLASSES = 43
FEATURES = len(semantic_map_keys_contract)
EMB = 1024


@lru_cache(maxsize=72)
def _load_arrays(doc_id):
  def _dp_fn(doc_id, suffix):
    return str(Path(work_dir) / 'datasets' / f'{doc_id}-datapoint-{suffix}.npy')

  embeddings = np.load(_dp_fn(doc_id, 'embeddings'))
  token_features = np.load(_dp_fn(doc_id, 'token_features'))
  semantic_map = np.load(_dp_fn(doc_id, 'semantic_map'))

  return embeddings, token_features, semantic_map


def make_xyw(doc_id: str, meta: DataFrame):
  row = meta.loc[doc_id]

  _subj = row['subject']
  subject_one_hot = ContractSubject.encode_1_hot()[_subj]

  embeddings, token_features, semantic_map = _load_arrays(doc_id)

  if embeddings.shape[0] != token_features.shape[0]:
    msg = f'{doc_id} embeddings.shape {embeddings.shape} is incompatible with token_features.shape {token_features.shape}'
    raise AssertionError(msg)

  if embeddings.shape[0] != semantic_map.shape[0]:
    msg = f'{doc_id} embeddings.shape {embeddings.shape} is incompatible with semantic_map.shape {semantic_map.shape}'
    raise AssertionError(msg)

  meta.at[doc_id, 'error'] = None
  return (
    (embeddings, token_features),
    (semantic_map, subject_one_hot),
    (row['sample_weight'], row['subject_weight']))


def validate_datapoint(id: str, meta: DataFrame):
  try:
    (emb, tok_f), (sm, subj), (sample_weight, subject_weight) = make_xyw(id, meta)
    if sm.shape[1] != len(semantic_map_keys_contract):
      mxs = f'semantic map shape is {sm.shape[1]}, expected is {len(semantic_map_keys_contract)} source={meta.at[id, "source"]}'
      raise ValueError(mxs)

  except Exception as e:
    raise ValueError(e)


def structure_detection_model_001(name, ctx: KerasTrainingContext = DEFAULT_TRAIN_CTX, trained=False):
  input_text_emb = Input(shape=[None, EMB], dtype='float32', name="input_text_emb")
  token_features = Input(shape=[None, TOKEN_FEATURES], dtype='float32', name="input_headlines_att")

  _out = LayerNormalization(epsilon=1e-6, name="ln_1e")(input_text_emb)
  token_features_n = LayerNormalization(epsilon=1e-6, name="ln_1t")(token_features)

  _out = Dropout(0.45, name="drops")(_out)  # small_drops_of_poison
  _out = concatenate([_out, token_features_n], axis=-1)
  _out = Conv1D(filters=FEATURES * 4, kernel_size=(2), padding='same', activation=None)(_out)
  _out = Conv1D(filters=FEATURES * 4, kernel_size=(4), padding='same', activation='relu', name='embedding_reduced')(
    _out)

  _out = Dropout(0.15)(_out)
  #   _out = BatchNormalization(name="bn_2")(_out)

  _out = LSTM(FEATURES * 4, return_sequences=True, activation="tanh")(_out)
  _out = LSTM(FEATURES, return_sequences=True, activation='tanh')(_out)
  #   _out = ReLU()(_out)

  model = Model(inputs=[input_text_emb, token_features], outputs=_out, name=name)

  model.compile(loss=sigmoid_focal_crossentropy, optimizer='Nadam',
                metrics=['mse', 'kullback_leibler_divergence', 'acc'])
  return model


def get_base_model(factory, ctx: KerasTrainingContext = DEFAULT_TRAIN_CTX, load_weights=True):
  model_001 = ctx.init_model(factory, trained=True, verbose=1, load_weights=load_weights)

  # BASE
  base_model = model_001.get_layer(name='embedding_reduced').output
  in1 = model_001.get_layer(name='input_text_emb').input
  in2 = model_001.get_layer(name='input_headlines_att').input

  return base_model, [in1, in2]


def uber_detection_model_001(name, ctx: KerasTrainingContext = DEFAULT_TRAIN_CTX, trained=False):
  """
  Evaluation:
  > 0.0030140 	loss
  > 0.0100294 	O1_tagging_loss
  > 0.0059756 	O2_subject_loss


  :param name:
  :return:
  """

  base_model, base_model_inputs = get_base_model(structure_detection_model_001, ctx=ctx, load_weights=not trained)

  _out_d = Dropout(0.1, name='alzheimer')(base_model)  # small_drops_of_poison
  _out = LSTM(FEATURES * 4, return_sequences=True, activation="tanh", name='paranoia')(_out_d)
  _out = LSTM(FEATURES, return_sequences=True, activation='tanh', name='O1_tagging_tanh')(_out)
  _out = ReLU(name='O1_tagging')(_out)

  # OUT 2: subject detection
  #
  pool_size = 2
  _out2 = MaxPooling1D(pool_size=pool_size, name='emotions')(_out_d)
  _out_mp = MaxPooling1D(pool_size=pool_size, name='insights')(_out)
  _out2 = concatenate([_out2, _out_mp], axis=-1, name='bipolar_disorder')
  _out2 = Bidirectional(LSTM(16, return_sequences=False, name='narcissism'), name='self_reflection')(_out2)

  _out2 = Dense(CLASSES, activation='softmax', name='O2_subject')(_out2)

  _losses = {
    "O1_tagging": sigmoid_focal_crossentropy,
    "O2_subject": "binary_crossentropy",
  }
  model = Model(inputs=base_model_inputs, outputs=[_out, _out2], name=name)
  model.compile(loss=_losses, optimizer='adam', metrics=metrics)
  return model


def uber_detection_model_003(name, ctx: KerasTrainingContext = DEFAULT_TRAIN_CTX, trained=False) -> Model:
  # BASE
  base_model, base_model_inputs = get_base_model(structure_detection_model_001, ctx=ctx, load_weights=not trained)
  # ---------------------

  _out_d = Dropout(0.5, name='alzheimer')(base_model)  # small_drops_of_poison
  _out = LSTM(FEATURES * 4, return_sequences=True, activation="tanh", name='paranoia')(_out_d)
  _out = LSTM(FEATURES, return_sequences=True, activation='tanh', name='O1_tagging_tanh')(_out)
  _out = ReLU(name='O1_tagging')(_out)

  # OUT 2: subject detection
  #
  pool_size = 2
  _out2 = MaxPooling1D(pool_size=pool_size, name='emotions')(_out_d)
  _out_mp = MaxPooling1D(pool_size=pool_size, name='insights')(_out)
  _out2 = concatenate([_out2, _out_mp], axis=-1, name='bipolar_disorder')
  _out2 = Dropout(0.3, name='alzheimer_3')(_out2)
  _out2 = Bidirectional(LSTM(16, return_sequences=False, name='narcissisism'), name='self_reflection')(_out2)

  _out2 = Dense(CLASSES, activation='softmax', name='O2_subject')(_out2)

  _losses = {
    "O1_tagging": sigmoid_focal_crossentropy,
    "O2_subject": "binary_crossentropy",
  }
  model = Model(inputs=base_model_inputs, outputs=[_out, _out2], name=name)
  model.compile(loss=_losses, optimizer='adam', metrics=metrics)
  return model


def uber_detection_model_005_1_1(name="uber_detection_model_005_1_1", ctx: KerasTrainingContext = DEFAULT_TRAIN_CTX,
                                 trained=False) -> Model:
  base_model, base_model_inputs = get_base_model(uber_detection_model_003, ctx=ctx, load_weights=False)

  # ---------------------

  _out_d = Dropout(0.35, name='alzheimer')(base_model)  # small_drops_of_poison
  _out = Bidirectional(LSTM(FEATURES * 4, return_sequences=True, name='paranoia'), name='self_reflection_4')(_out_d)
  _out = Dropout(0.3, name='alzheimer_11')(_out)
  _out_l = LSTM(FEATURES, return_sequences=True, activation='tanh', name='O1_tagging_tanh')(_out)

  # OUT 2: subject detection
  pool_size = 2
  emotions = MaxPooling1D(pool_size=pool_size, name='emotions')(_out_d)
  insights = MaxPooling1D(pool_size=pool_size, name='insights')(_out_l)
  _out2 = concatenate([emotions, insights], axis=-1, name='bipolar_disorder')
  _out2 = Dropout(0.3, name='alzheimer_3')(_out2)
  _out2 = Bidirectional(LSTM(32, return_sequences=False, name='narcissisism'), name='self_reflection_2')(_out2)
  _out2 = Dropout(0.1, name='alzheimer_1')(_out2)

  _out2 = Dense(CLASSES, activation='softmax', name='O2_subject')(_out2)

  _out = LeakyReLU(name='O1_tagging')(_out_l)
  model = Model(inputs=base_model_inputs, outputs=[_out, _out2], name=name)
  model.compile(loss=losses, optimizer='Nadam', metrics=metrics)
  return model


def uber_detection_model_006(name, ctx: KerasTrainingContext = DEFAULT_TRAIN_CTX, trained=False):
  input_text_emb = Input(shape=[None, EMB], dtype='float32', name="input_text_emb")
  input_token_features = Input(shape=[None, TOKEN_FEATURES], dtype='float32', name="input_token_features")

  base_model_inputs = [input_text_emb, input_token_features]

  _out = Dropout(0.45, name="drops")(input_text_emb)  # small_drops_of_poison
  _out = Conv1D(filters=FEATURES * 5, kernel_size=(3), padding='same', activation=None)(_out)
  _out = concatenate([_out, input_token_features], axis=-1)
  _out = Conv1D(filters=FEATURES * 5, kernel_size=(4),
                padding='same', activation='relu',
                name='embedding_reduced')(_out)

  # ---------------------

  _out_d = Dropout(0.15, name='alzheimer')(_out)  # small_drops_of_poison
  _out = Bidirectional(LSTM(FEATURES * 2, return_sequences=True, name='paranoia'), name='self_reflection_1')(_out_d)
  _out = Dropout(0.1, name='alzheimer_11')(_out)
  _out = LSTM(FEATURES, return_sequences=True, activation='tanh', name='O1_tagging')(_out)

  # OUT 2: subject detection
  pool_size = 2
  emotions = MaxPooling1D(pool_size=pool_size, name='emotions')(_out_d)
  insights = MaxPooling1D(pool_size=pool_size, name='insights')(_out)
  _out2 = concatenate([emotions, insights], axis=-1, name='bipolar_disorder')
  _out2 = Dropout(0.3, name='alzheimer_3')(_out2)
  _out2 = Bidirectional(LSTM(16, return_sequences=False, name='narcissisism'), name='self_reflection_2')(_out2)
  _out2 = Dropout(0.1, name='alzheimer_1')(_out2)

  _out2 = Dense(CLASSES, activation='softmax', name='O2_subject')(_out2)

  model = Model(inputs=base_model_inputs, outputs=[_out, _out2], name=name)
  model.compile(loss=losses, optimizer='Nadam', metrics=metrics)
  return model


@dataclass
class Config:
  # MAX_LEN = 256
  # BATCH_SIZE = 32
  LR = 0.001

  EMBED_DIM = EMB
  NUM_HEAD = 4  # used in bert model
  FF_DIM = 128  # used in bert model
  NUM_LAYERS = 1


config = Config()


def bert_module(query, key, value, i, height, key_dim_base=config.EMBED_DIM):
  # Multi headed self-attention
  attention_output = layers.MultiHeadAttention(
    num_heads=config.NUM_HEAD,
    key_dim=key_dim_base // config.NUM_HEAD,
    name="encoder_{}/multiheadattention".format(i),
  )(query, key, value)
  attention_output = layers.Dropout(0.1, name="encoder_{}/att_dropout".format(i))(
    attention_output
  )
  attention_output = layers.LayerNormalization(
    epsilon=1e-6, name=f"encoder_{i}/att_layernormalization"
  )(query + attention_output)

  # Feed-forward layer
  ffn = keras.Sequential(
    [
      layers.Dense(config.FF_DIM, activation="relu"),
      layers.Dense(height),
    ],
    name=f"encoder_{i}/ffn",
  )
  ffn_output = ffn(attention_output)
  ffn_output = layers.Dropout(0.1, name=f"encoder_{i}/ffn_dropout")(
    ffn_output
  )
  sequence_output = layers.LayerNormalization(
    epsilon=1e-6, name=f"encoder_{i}/ffn_layernormalization"
  )(attention_output + ffn_output)
  return sequence_output


class ThresholdLayer(layers.Layer):
  def __init__(self, **kwargs):
    super(ThresholdLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    self.kernel = self.add_weight(name="threshold", shape=(1,), initializer="uniform",
                                  trainable=True)
    super(ThresholdLayer, self).build(input_shape)

  def call(self, x, *args, **kwargs):
    return keras.backend.sigmoid(100 * (x - self.kernel))

  def compute_output_shape(self, input_shape):
    return input_shape


SEQUENCE_AXIS = -2


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
    self.sequence_length = int(sequence_length)
    self.initializer = keras.initializers.get(initializer)

  def get_config(self):
    config = super().get_config()
    config.update(
      {
        "sequence_length": self.sequence_length,
        "initializer": keras.initializers.serialize(self.initializer),
      }
    )
    return config

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
    config = super().get_config()
    config.update(
      {
        "max_wavelength": self.max_wavelength,
      }
    )
    return config


def make_att_model(name='make_att_model', ctx: KerasTrainingContext = DEFAULT_TRAIN_CTX, trained=False):
  input_text_emb = layers.Input(shape=[None, config.EMBED_DIM], dtype='float32', name="input_text_emb")
  _out = layers.BatchNormalization(name="bn1")(input_text_emb)
  _out = layers.Dropout(0.2, name="drops")(_out)  # small_drops_of_poison

  token_features = layers.Input(shape=[None, TOKEN_FEATURES], dtype='float32', name="token_features")
  token_features_n = layers.BatchNormalization(name="bn2")(token_features)

  _out = layers.concatenate([input_text_emb, token_features_n], axis=-1, name='rmb_plus_tokens')

  for i in range(config.NUM_LAYERS):
    _out = bert_module(_out, _out, _out, i, height=config.EMBED_DIM + TOKEN_FEATURES)

  _out = layers.BatchNormalization(name="bn1")(_out)
  _out = layers.LSTM(FEATURES, return_sequences=True, activation='tanh', name='O1_tagging_tanh')(_out)
  #   _out1 = layers.ReLU(name='O1_tagging')(_out)
  _out1 = ThresholdLayer(name='O1_tagging')(_out)

  #   _out = Conv1D(filters=FEATURES * 4, kernel_size=(2), padding='same', activation='relu' , name='embedding_reduced')(_out)
  _out = layers.Bidirectional(layers.LSTM(16, return_sequences=False, name='narcissisism'), name='embedding_reduced')(
    _out)
  _out = layers.BatchNormalization(name="bn_bi_2")(_out)
  _out = layers.Dropout(0.1, name='forgetting')(_out)

  _out2 = layers.Dense(CLASSES, activation='softmax', name='O2_subject')(_out)

  base_model_inputs = [input_text_emb, token_features]
  model = Model(inputs=base_model_inputs, outputs=[_out1, _out2], name=name)
  model.compile(loss=losses, optimizer='Adam', metrics=metrics)
  return model


def make_att_model_02(name='make_att_model_02', ctx: KerasTrainingContext = DEFAULT_TRAIN_CTX, trained=False) -> Model:
  # ---------------------
  input_text_emb = layers.Input(shape=[None, EMB], dtype='float32', name="input_text_emb")
  input_text_emb_n = layers.Dropout(0.15, name='alzheimer_001')(input_text_emb)
  input_text_emb_n = layers.LayerNormalization(epsilon=1e-6, name="ln_1e")(input_text_emb_n)

  token_features = layers.Input(shape=[None, TOKEN_FEATURES], dtype='float32', name="input_token_features")

  token_features_n = layers.LayerNormalization(epsilon=1e-6, name="ln_1t")(token_features)
  # token_features_n = token_features_n + PositionEmbedding(name='token_pos_emb')(token_features_n)

  # reducing size of embedding
  _out = layers.concatenate([input_text_emb_n, token_features_n], axis=-1)
  _out = layers.Conv1D(filters=FEATURES * 4, kernel_size=(2), padding='same', activation=None)(_out)
  _emb__len = FEATURES * 4
  embedding_reduced = layers.Conv1D(filters=_emb__len, kernel_size=(4), padding='same', activation='relu',
                                    name='embedding_reduced')(
    _out)

  _out = layers.BatchNormalization(name="norm_embedding_reduced")(embedding_reduced)
  #   _pos_emb = PositionEmbedding(sequence_length=MAX_LEN)(_out)
  _pos_emb = SinePositionEncoding(name='sine_position')(_out)

  _out = _pos_emb + _out
  _out = layers.Dropout(0.15, name='alzheimer_002')(_out)
  #   _out = _out + PositionEmbedding(embed_dim=_emb__len, name='position_emb')(_out)

  for i in range(config.NUM_LAYERS):
    _out = bert_module(_out, _out, _out, i, height=_emb__len)

  bert_out = _out
  bert_out = layers.BatchNormalization(name="norm_bert_out")(bert_out)

  if True:
    # branch 1
    # _out = layers.LSTM(FEATURES, return_sequences=True, activation='tanh', name='O1_tagging_tanh')(bert_out)
    _out = layers.Bidirectional(layers.LSTM(FEATURES // 2, return_sequences=True, name='narcissisism1', activation='tanh'), name='O1_tagging_tanh')(bert_out)
    _out1 = ThresholdLayer(name='O1_tagging')(_out)

  if True:
    _out2 = layers.Bidirectional(layers.LSTM(32, return_sequences=False, name='narcissisism2'),
                                 name='self_reflection_2')(bert_out)
    _out2 = layers.Dropout(0.15, name='alzheimer_1')(_out2)

    _out2 = layers.Dense(CLASSES, activation='softmax', name='O2_subject')(_out2)

  base_model_inputs = [input_text_emb, token_features]
  model = Model(inputs=base_model_inputs, outputs=[_out1, _out2], name=name)
  model.compile(loss=losses, optimizer='Adam', metrics=metrics)

  return model



def make_att_model_03(name='make_att_model_03', ctx = None, trained=False):
  input_text_emb = layers.Input(shape=[None, config.EMBED_DIM], dtype='float32', name="input_text_emb")
  input_text_emb_n = layers.LayerNormalization(epsilon=1e-6, name="input_text_emb_norm")(input_text_emb)
  # _out = layers.BatchNormalization(name="bn1")(input_text_emb)
  # _out = layers.Dropout(0.2, name="drops")(_out)  # small_drops_of_poison


  token_features = layers.Input(shape=[None, TOKEN_FEATURES], dtype='float32', name="token_features")
  token_features_n = layers.LayerNormalization(epsilon=1e-6, name="token_features_norm")(token_features)
  
  # token_features_n = layers.BatchNormalization(name="bn2")(token_features)

  _lstm_height = 128
  _out = layers.concatenate([input_text_emb_n, token_features_n], axis=-1, name='rmb_plus_tokens')
  _out = layers.Bidirectional(layers.LSTM(_lstm_height, return_sequences=True, name='narcissisism1', activation='tanh'), name='embedding_reduced')(_out)
  _out = layers.Dropout(0.2, name='amnesia')(_out)
  _out = layers.BatchNormalization(name="bn1")(_out)
    

  _bert = _out
  for i in range(2):
    _bert = bert_module(_bert, _bert, _bert, i, height=_lstm_height*2, key_dim_base=_lstm_height*2)

  
  _bert = layers.BatchNormalization(name="bn2")(_bert)

  _out = _bert
  _out = layers.LSTM(FEATURES, return_sequences=True, activation='tanh', name='O1_tagging_tanh')(_out)
  #   _out1 = layers.ReLU(name='O1_tagging')(_out)
  _out1 = ThresholdLayer(name='O1_tagging')(_out)

  #   _out = Conv1D(filters=FEATURES * 4, kernel_size=(2), padding='same', activation='relu' , name='embedding_reduced')(_out)
  _out = layers.Bidirectional(layers.LSTM(16, return_sequences=False, name='narcissisism2', activation='tanh'), name='some')(_bert)
  # _out = layers.BatchNormalization(name="bn_bi_2")(_out)
  # 

  _out2 = layers.Dense(CLASSES, activation='softmax', name='O2_subject')(_out)

  base_model_inputs = [input_text_emb, token_features]
  model = Model(inputs=base_model_inputs, outputs=[_out1, _out2], name=name)
  model.compile(loss=losses, optimizer='Adam', metrics=metrics)
  return model



make_att_model = make_att_model_03
###-------------------------


def get_amount(attr_tree):
  _value_tag = attr_tree.get('price')
  amount = None
  if _value_tag is not None:
    amount = _value_tag.get('amount_netto')
    if amount is None:
      amount = _value_tag.get('amount_brutto')
    if amount is None:
      amount = _value_tag.get('amount')
  return amount


# --------------------------

def fix_contract_number_span(span: [], textmap):
  if span is not None:
    span = [span[0], span[1]]  # //typesafety
    for i in range(span[0], span[1]):
      t = textmap[i]
      t = t.strip('_')
      t = t.strip().lstrip('№').lstrip().lstrip(':').lstrip('N ').lstrip().rstrip('.')
      if t == '':
        span[0] = i + 1
    for i in range(span[1], span[0]):
      t = textmap[i]
      t = t.strip('_')
      t = t.strip().lstrip('№').lstrip().lstrip(':').lstrip('N ').lstrip().rstrip('.')
      if t == '':
        span[1] = i - 1

  #     if span[1]-span[0] == 0:
  #       return None

  return span


def get_semantic_map_new(doc) -> DataFrame:
  _len = len(doc)
  df = DataFrame()

  # init datatable with zeros
  for sl in semantic_map_keys_contract:
    df[sl] = np.zeros(_len)

  attr_tree = doc.get_attributes_tree()

  def add_span_vectors(_name, span):
    bn = _name + "-begin"
    en = _name + "-end"
    if not span is None:
      df[bn][span[0]:span[1]] = 1.
      df[en][span[1]] = 1.

  # Headers
  headers = doc.analysis['headers']
  for h in headers:
    add_span_vectors('headline', h['span'])

  for n in t_semantic_map_keys_common[1:]:  # 1: == skip headers
    span = attr_tree.get(n, {}).get('span')
    if n == 'number':
      #         print(f'number: {[doc.get_tokens_map_unchaged().text_range(span)]}')
      span1 = fix_contract_number_span(span, doc.get_tokens_map_unchaged())
      if span != span1:
        print(
          f'fixed number: {[doc.get_tokens_map_unchaged().text_range(span)]} -->  {[doc.get_tokens_map_unchaged().text_range(span1)]}')
      span = span1
    if span:
      add_span_vectors(n, span)

  # Orgs:
  for org in attr_tree.get('orgs', []):  # org number (index)
    for org_part_key in t_semantic_map_keys_org:

      org_part = org.get(org_part_key.replace('org-', ''), {})
      if org_part:
        span = org_part.get('span', None)
        add_span_vectors(org_part_key, span)

  _value_tag = attr_tree.get('price', {})

  if _value_tag is not None:
    add_span_vectors("value", _value_tag.get('span'))
    amount = get_amount(attr_tree)
    if amount:
      add_span_vectors('amount', amount.get('span'))

    #     print('_value_tag=', _value_tag)
    #     print('amount=', amount)
    for n in t_semantic_map_keys_price:
      _value_tag_part = _value_tag.get(n)
      #       print('n=', n)
      #       print('_value_tag_part=', _value_tag_part)
      if _value_tag_part:
        add_span_vectors(n, _value_tag_part.get('span'))

  return df[semantic_map_keys_contract]


if __name__ == '__main__':
  # print(FEATURES)
  model = make_att_model()
  model.summary()
