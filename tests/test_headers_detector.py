#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import os
import pickle
import unittest

from analyser.headers_detector import doc_features, load_model
from analyser.legal_docs import LegalDocument


def print_predictions(contract, predictions, body_lines_ranges):
  headlines_cnt = 0
  for i in range(len(predictions)):
    if predictions[i] > 0.1:
      headlines_cnt += 1
      # print(f'{predictions[i]} \t {i}\t🎖{contract.tokens_map.text_range(body_lines_ranges[i])}❗')
  return headlines_cnt


class TestHeaderDetector(unittest.TestCase):

  @unittest.skip("headers detector should be retrained")
  def test_doc_features(self):
    with open(os.path.join(os.path.dirname(__file__), '2. Договор по благ-ти Радуга.docx.pickle'), 'rb') as handle:
      contract: LegalDocument = pickle.load(handle)

    features, _ = doc_features(contract.tokens_map)
    self.assertEqual(27, len(features))
    print(features[0])
    pass

  def test_doc_features_predict(self):
    with open(os.path.join(os.path.dirname(__file__), '2. Договор по благ-ти Радуга.docx.pickle'), 'rb') as handle:
      doc: LegalDocument = pickle.load(handle)

    features, body_lines_ranges = doc_features(doc.tokens_map)

    model = load_model()
    predictions = model.predict(features)

    headlines_cnt = print_predictions(doc, predictions, body_lines_ranges)
    expected_p = len(doc.paragraphs)
    eps = 5
    self.assertLess(headlines_cnt, expected_p + eps)
    self.assertGreater(headlines_cnt, expected_p - eps)

  @unittest.skip("does not work for protocols :(")
  def test_doc_features_predict_protocol(self):
    with open(os.path.join(os.path.dirname(__file__), 'Протокол_СД_ 3.docx.pickle'), 'rb') as handle:
      doc: LegalDocument = pickle.load(handle)

    features, body_lines_ranges = doc_features(doc.tokens_map)

    model = load_model()
    predictions = model.predict(features)

    headlines_cnt = print_predictions(doc, predictions, body_lines_ranges)
    expected_p = len(doc.paragraphs)
    eps = 5
    self.assertLess(headlines_cnt, expected_p + eps)
    self.assertGreater(headlines_cnt, expected_p - eps)




  def test_doc_features_predict_2(self):
    with open(os.path.join(os.path.dirname(__file__), 'Договор 8.docx.pickle'), 'rb') as handle:
      contract: LegalDocument = pickle.load(handle)



    features, body_lines_ranges = doc_features(contract.tokens_map)

    model = load_model()
    predictions = model.predict(features)

    headlines_cnt = print_predictions(contract, predictions, body_lines_ranges)

    expected_p = len(contract.paragraphs)
    eps = 5
    self.assertLess(headlines_cnt, expected_p+eps)
    self.assertGreater(headlines_cnt, expected_p-eps)

  # @unittest.skip("headers detector should be retrained")
  def test_doc_features_predict_3(self):
    with open(os.path.join(os.path.dirname(__file__), 'Договор _2_.docx.pickle'), 'rb') as handle:
      contract: LegalDocument = pickle.load(handle)



    features, body_lines_ranges = doc_features(contract.tokens_map)

    model = load_model()
    predictions = model.predict(features)

    headlines_cnt = print_predictions(contract, predictions, body_lines_ranges)
    expected_p = len(contract.paragraphs)
    eps = 5
    self.assertLess(headlines_cnt, expected_p + eps)
    self.assertGreater(headlines_cnt, expected_p - eps)




unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
