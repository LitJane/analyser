# see  /notebooks/TF_subjects.ipynb
import warnings

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Bidirectional, Input, Dropout
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.models import Model

from analyser.headers_detector import get_tokens_features
from analyser.hyperparams import models_path
from analyser.ml_tools import FixedVector
from analyser.structures import ContractSubject
from tf_support.super_contract_model import make_att_model, semantic_map_keys_contract
from tf_support.tools import KerasTrainingContext
from trainsets.trainset_tools import SubjectTrainsetManager

VALIDATION_SET_PROPORTION = 0.25

EMB = 1024  # embedding dimentionality


def decode_subj_prediction(result: FixedVector) -> (ContractSubject, float, int):
  max_i = result.argmax()
  predicted_subj_name = ContractSubject(max_i)
  confidence = float(result[max_i])
  return predicted_subj_name, confidence, max_i


def nn_predict(umodel, doc):
  embeddings = doc.embeddings
  token_features = get_tokens_features(doc.tokens)
  prediction = umodel.predict(x=[np.expand_dims(embeddings, axis=0), np.expand_dims(token_features, axis=0)],
                              batch_size=1)

  semantic_map = pd.DataFrame(prediction[0][0], columns=semantic_map_keys_contract)
  return semantic_map, prediction[1][0]


predict_subject = nn_predict


def set_conv_bi_LSTM_dropouts_training_params(dataset_manager: SubjectTrainsetManager):
  dataset_manager.noisy_samples_amount = 0.75
  dataset_manager.outliers_percent = 0.1
  dataset_manager.noise_amount = 0.05


def conv_bi_LSTM_dropouts_binary(name="new_model"):
  warnings.warn('not the best model, use uber', DeprecationWarning)
  CLASSES = 43
  input_text = Input(shape=[None, EMB], dtype='float32', name="input_text_emb")

  _out = input_text
  _out = Dropout(0.1, name="drops")(_out)
  _out = Conv1D(filters=16, kernel_size=(8), padding='same', activation='relu')(_out)
  _out = MaxPooling1D(pool_size=2)(_out)
  _out = Bidirectional(LSTM(16, return_sequences=False))(_out)
  _out = Dense(CLASSES, activation='softmax')(_out)

  model = Model(inputs=[input_text], outputs=_out, name=name)
  model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy', 'categorical_accuracy'])

  return model


def conv_biLSTM_binary_dropouts05(name="new_model"):
  warnings.warn('not the best model, use uber', DeprecationWarning)
  CLASSES = 43
  input_text = Input(shape=[None, EMB], dtype='float32', name="input_text_emb")

  _out = input_text
  _out = Dropout(0.5, name="drops")(_out)
  _out = Conv1D(filters=16, kernel_size=(8), padding='same', activation='relu')(_out)
  _out = MaxPooling1D(pool_size=2)(_out)
  _out = Bidirectional(LSTM(16, return_sequences=False))(_out)
  _out = Dense(CLASSES, activation='softmax')(_out)

  model = Model(inputs=[input_text], outputs=_out, name=name)
  model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy', 'categorical_accuracy'])

  return model


def conv_bi_LSTM_dropouts(name="new_model") -> Model:
  warnings.warn('not the best model, use uber', DeprecationWarning)
  '''
  Epoch 20
  loss: 0.0206 - acc: 1.0000 - val_loss: 0.3490 - val_acc: 0.9417


  see checkpoints/conv_bi_LSTM_dropouts
  :param name: name of the model
  :return: compiled TF model
  '''

  CLASSES = 43
  input_text = Input(shape=[None, EMB], dtype='float32', name="input_text_emb")

  _out = input_text
  _out = Dropout(0.1, name="drops")(_out)
  _out = Conv1D(filters=16, kernel_size=(8), padding='same', activation='relu')(_out)
  _out = MaxPooling1D(pool_size=2)(_out)
  _out = Bidirectional(LSTM(16, return_sequences=False))(_out)
  _out = Dense(CLASSES, activation='softmax')(_out)

  model = Model(inputs=[input_text], outputs=_out, name=name)
  model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy', 'categorical_accuracy'])

  return model


def load_subject_detection_trained_model() -> Model:
  ctx = KerasTrainingContext(models_path)

  final_model = ctx.init_model(make_att_model, trained=True, trainable=False, verbose=10)

  # print(final_model.name)
  # final_model.summary()

  return final_model
