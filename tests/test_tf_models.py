#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

import unittest

import numpy as np
from tensorflow import keras
from tensorflow.keras import Model

from analyser.ml_tools import Embeddings
from tests.test_utilits import load_json_sample
from tf_support.embedder_elmo import ElmoEmbedder
from tf_support.super_contract_model import uber_detection_model_005_1_1, uber_detection_model_003, \
  structure_detection_model_001
from tf_support.tools import KerasTrainingContext
from trainsets.retrain_contract_uber_model import _get_semantic_map, DbJsonDoc


class TestLoadTfModel(unittest.TestCase):
  def test_tf_version(self):
    print(keras.__version__)

  def test_embedder(self):
    embedder = ElmoEmbedder.get_instance()
    e: Embeddings = embedder.embedd_tokens([['мама', 'мыла', 'Раму', 'а', 'Рама', 'мыл', 'Шиву']])
    # just expecting NO failure
    print(e)

  def test_get_semantic_map(self):
    json_dict = load_json_sample('contract_db_1.json')

    sm = _get_semantic_map(DbJsonDoc(json_dict))
    print(sm.shape)

  def test_resave_models_h5(self):
    ctx = KerasTrainingContext()
    ctx.resave_model_h5(structure_detection_model_001)
    ctx.resave_model_h5(uber_detection_model_003)
    ctx.resave_model_h5(uber_detection_model_005_1_1)

  def test_load_uber_model_005(self):
    ctx = KerasTrainingContext()
    model_factory_fn = uber_detection_model_005_1_1
    model: Model = ctx.init_model(model_factory_fn, verbose=2)
    self.assertIsNotNone(model)

    fake_emb = np.zeros((1, 100, 1024))
    fake_lab = np.zeros((1, 100, 15))

    prediction = model.predict([fake_emb, fake_lab])
    self.assertEqual(2, len(prediction))


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
