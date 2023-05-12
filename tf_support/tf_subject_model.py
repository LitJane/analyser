# see  /notebooks/TF_subjects.ipynb

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model

from analyser.headers_detector import get_tokens_features
from analyser.hyperparams import models_path
from analyser.ml_tools import FixedVector
from analyser.structures import ContractSubject
from tf_support.super_contract_model import make_att_model, semantic_map_keys_contract
from tf_support.tools import KerasTrainingContext

EMB = 1024  # embedding dimentionality


def decode_subj_prediction(result: FixedVector) -> (ContractSubject, float, int):
  max_i = result.argmax()
  try:
    predicted_subj_name = ContractSubject(max_i)
    confidence = float(result[max_i])
    return predicted_subj_name, confidence, max_i
  except ValueError:
    return ContractSubject(0), 0.0, 0


def nn_predict(umodel, doc):
  embeddings = doc.embeddings
  token_features = get_tokens_features(doc.tokens)
  prediction = umodel.predict(x=[np.expand_dims(embeddings, axis=0), np.expand_dims(token_features, axis=0)],
                              batch_size=1)

  semantic_map = pd.DataFrame(prediction[0][0], columns=semantic_map_keys_contract)
  return semantic_map, prediction[1][0]


predict_subject = nn_predict


def load_subject_detection_trained_model() -> Model:
  ctx = KerasTrainingContext(models_path)

  final_model = ctx.init_model(make_att_model, trained=True, trainable=False, verbose=10)

  return final_model
