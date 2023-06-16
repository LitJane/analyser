import json
import os
import pickle

import numpy as np
from bson import json_util

import gpn_config
from analyser.contract_parser import ContractDocument
from analyser.embedding_tools import AbstractEmbedder, Embeddings
from analyser.text_tools import Tokens

NO_DB = gpn_config.config.get('GPN_DB_HOST', None) is None
NO_DB_ERR_MSG = "requires GPN_DB_HOST to be configured"


def load_json_sample(fn: str) -> dict:
  pth = os.path.dirname(__file__)
  with open(os.path.join(pth, fn), 'rb') as handle:
    data = json.load(handle, object_hook=json_util.object_hook)

  return data


def get_a_contract() -> ContractDocument:
  pth = os.path.dirname(__file__)
  with open(pth + '/2. Договор по благ-ти Радуга.docx.pickle', 'rb') as handle:
    doc = pickle.load(handle)

  return doc


class FakeEmbedder(AbstractEmbedder):

  def __init__(self, default_point):
    self.default_point = default_point

  def embedd_tokens(self, tokens: Tokens) -> Embeddings:
    return self.embedd_tokenized_text([tokens], [len(tokens)])[0]

  def embedd_strings(self, strings: Tokens) -> Embeddings:
    ret = self.embedd_tokens(strings)
    return ret

  def embedd_tokenized_text(self, tokenized_sentences_list, lens):
    tensor = []
    for sent in tokenized_sentences_list:
      sentense_emb = [self.default_point] * len(sent)
      tensor.append(sentense_emb)

    return np.array(tensor)
