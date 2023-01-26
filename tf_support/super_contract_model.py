from functools import lru_cache
from pathlib import Path

import numpy as np
from pandas import DataFrame
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, Dropout, LSTM, Bidirectional, Dense, MaxPooling1D, ReLU
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

metrics = ['kullback_leibler_divergence', 'mse', 'binary_crossentropy']

losses = {
  "O1_tagging": "binary_crossentropy",
  "O2_subject": "binary_crossentropy",
}

# seq_labels_contract = seq_labels_contract_level_1 + seq_labels_dn + seq_labels_org_1 + seq_labels_org_2 + seq_labels_val
# seq_labels_contract_swap_orgs = seq_labels_contract_level_1 + seq_labels_dn + seq_labels_org_2 + seq_labels_org_1 + seq_labels_val

semantic_map_keys = [
  'headline',
  'subject',
  'date',
  'number',
  'org-name',
  'org-alias',
  'org-type'
]

semantic_map_keys += ['amount', 'amount_brutto', 'amount_netto', 'vat', 'sign', 'currency', 'vat_unit', 'value']

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

  _out = Dropout(0.45, name="drops")(input_text_emb)  # small_drops_of_poison
  _out = concatenate([_out, token_features], axis=-1)
  _out = Conv1D(filters=FEATURES * 4, kernel_size=(2), padding='same', activation=None)(_out)
  _out = Conv1D(filters=FEATURES * 4, kernel_size=(4), padding='same', activation='relu', name='embedding_reduced')(
    _out)

  _out = Dropout(0.15)(_out)

  _out = LSTM(FEATURES * 4, return_sequences=True, activation="tanh")(_out)
  _out = LSTM(FEATURES, return_sequences=True, activation='tanh')(_out)
  _out = ReLU()(_out)
    
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


def uber_detection_model_005_1_1(name, ctx: KerasTrainingContext = DEFAULT_TRAIN_CTX, trained=False) -> Model:
  base_model, base_model_inputs = get_base_model(uber_detection_model_003, ctx=ctx, load_weights=False)

  # ---------------------

  _out_d = Dropout(0.35, name='alzheimer')(base_model)  # small_drops_of_poison
  _out = Bidirectional(LSTM(FEATURES * 2, return_sequences=True, name='paranoia'), name='self_reflection_1')(_out_d)
  _out = Dropout(0.5, name='alzheimer_11')(_out)
  _out = LSTM(FEATURES, return_sequences=True, activation='tanh', name='O1_tagging_tanh')(_out)
  _out = ReLU(name='O1_tagging')(_out)

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


