#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

import unittest

import numpy as np
from tensorflow import keras
from tensorflow.keras import Model

from analyser.ml_tools import Embeddings
from tf_support.embedder_elmo import ElmoEmbedder
from tf_support.super_contract_model import make_att_model
from tf_support.tools import KerasTrainingContext


class TestLoadTfModel(unittest.TestCase):
  def test_tf_version(self):
    print(keras.__version__)

  def test_embedder(self):
    embedder = ElmoEmbedder.get_instance()
    e: Embeddings = embedder.embedd_tokens([['мама', 'мыла', 'Раму', 'а', 'Рама', 'мыл', 'Шиву']])
    # just expecting NO failure
    print(e)

  def test_load_model(self):
    ctx = KerasTrainingContext()
    model_factory_fn = make_att_model
    model: Model = ctx.init_model(model_factory_fn, verbose=2)
    self.assertIsNotNone(model)

    fake_emb = np.zeros((1, 100, 1024))
    fake_lab = np.zeros((1, 100, 15))

    prediction = model.predict([fake_emb, fake_lab])
    self.assertEqual(2, len(prediction))


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
