#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

from analyser.headers_detector import line_features, model_path
from analyser.hyperparams import HyperParameters
from analyser.legal_docs import PARAGRAPH_DELIMITER, LegalDocument
from integration.db import get_mongodb_connection
from integration.word_document_parser import WordDocParser, join_paragraphs

MAX_DOCS = 500  # 0.0153 degrees.


def doc_line_features(doc: LegalDocument) -> []:
  tmap = doc.tokens_map
  _features = []
  ln: int = 0
  _prev_features = None
  for p in doc.paragraphs:

    header_tokens = tmap[p.header.slice]
    # print('☢️', header_tokens)

    header_features = line_features(tmap, p.header.span, ln, _prev_features)
    if len(header_tokens) == 1 and header_tokens[0] == '\n':
      header_features['actual'] = 0.0
    else:
      header_features['actual'] = 1.0

    _features.append(header_features)
    _prev_features = header_features.copy()
    ln += 1
    # --
    bodymap = tmap.slice(p.body.slice)
    body_lines_ranges = bodymap.split_spans(PARAGRAPH_DELIMITER, add_delimiter=True)
    # line_number:int=0
    for line_span in body_lines_ranges:
      # line_tokens = bodymap.tokens_by_range(line_span)

      body_features = line_features(bodymap, line_span, ln, _prev_features)
      body_features['actual'] = 0
      _features.append(body_features)
      _prev_features = body_features.copy()
      ln += 1

  return _features


if __name__ == '__main__':

  features_dicts = []
  count = 0

  db = get_mongodb_connection()
  criterion = {
    'version': WordDocParser.version
  }

  res = db['legaldocs'].find(criterion)

  for resp in res:
    for d in resp['documents']:
      doctype = d['documentType']
      if doctype in ('CONTRACT', 'PROTOCOL', 'CHARTER'):
        print(resp['_id'])
        legal_doc = join_paragraphs(d, resp['_id'])
        _doc_features = doc_line_features(legal_doc)
        features_dicts += _doc_features

        count += 1

    if count > MAX_DOCS:
      break

  featuresX_data = pd.DataFrame.from_records(features_dicts)
  print(featuresX_data.head())

  labels = np.array(featuresX_data['actual'])
  # Remove the labels from the features
  # axis 1 refers to the columns
  features = featuresX_data.drop('actual', axis=1)
  # Saving feature names for later use
  feature_list = list(features.columns)
  # Convert to numpy array
  features = np.array(features)

  # Split the data into training and testing sets
  train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                              random_state=42)
  print('Training Features Shape:', train_features.shape)
  print('Training Labels Shape:', train_labels.shape)
  print('Testing Features Shape:', test_features.shape)
  print('Testing Labels Shape:', test_labels.shape)

  if HyperParameters.headers_detector_use_regressor:
    model_class = RandomForestRegressor
  else:
    model_class = RandomForestClassifier

  # rf = RandomForestRegressor(n_estimators=150, random_state=42, min_samples_split=8)
  rf_class = model_class(n_estimators=150, random_state=42, min_samples_split=8)
  # Train the model
  rf_class.fit(train_features, train_labels)

  # Use the forest's predict method on the test data
  predictions = rf_class.predict(test_features)
  # # Calculate the absolute errors
  errors = abs(predictions - test_labels)
  # # Print out the mean absolute error (mae)
  print('Mean Absolute Error:', round(np.mean(errors), 4), 'degrees.')

  # dump(rf, os.path.join(models_path, 'rf_headers_detector_model.joblib'))
  dump(rf_class, model_path)
