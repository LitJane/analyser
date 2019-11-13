#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
import os
import sys
import traceback

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from headers_detector import line_features
from hyperparams import models_path
from integration.db import get_mongodb_connection
from integration.word_document_parser import WordDocParser, join_paragraphs
from legal_docs import PARAGRAPH_DELIMITER

files_dirs = [
  # '/Users/artem/Downloads/Telegram Desktop/X0/',
              '/Users/artem/work/nemo/goil/IN/',
              '/Users/artem/Google Drive/GazpromOil/Protocols', '/Users/artem/Google Drive/GazpromOil/Contracts',
              '/Users/artem/Google Drive/GazpromOil/Charters']
# MAX_DOCxS = 50 #0.0442
# MAX_DOCS = 150 #0.0418
MAX_DOCS = 500 #  0.0153 degrees.
from random import shuffle


def doc_line_features(contract) -> []:
  tmap = contract.tokens_map
  features = []
  ln = 0
  _prev_features = None
  for p in contract.paragraphs:

    header_tokens = tmap[p.header.slice]
    header_features = line_features(tmap, p.header.span, ln, _prev_features)

    header_features['actual'] = 1.0
    print('☢️', header_tokens)
    features.append(header_features)
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
      features.append(body_features)
      _prev_features = body_features.copy()
      ln += 1

  return features


def read_all_contracts():
  db = get_mongodb_connection()
  collection = db['legaldocs']

  wp = WordDocParser()
  filenames=[]
  for dir in files_dirs:
    filenames += wp.list_filenames(dir)

  shuffle(filenames)
  print(filenames)

  cnt = 0
  failures = 0
  _version = wp.version
  for fn in filenames:

    shortfn = fn.split('/')[-1]
    pth = '/'.join(fn.split('/')[5:-1])
    _doc_id = pth + '/' + shortfn

    cnt += 1
    print(cnt, fn)

    res = collection.find_one({"_id": _doc_id, 'version': _version})
    if res is None:
      try:
        # parse and save to DB
        res = wp.read_doc(fn)

        res['short_filename'] = shortfn
        res['path'] = pth
        res['_id'] = _doc_id
        res['version'] = _version

        collection.delete_one({'_id': _doc_id})
        collection.insert_one(res)

        yield res

      except Exception:
        print(f"{fn}\nException in WordDocParser code:")
        traceback.print_exc(file=sys.stdout)
        failures += 1

    else:
      yield res
      # print(cnt, res["documentDate"], res["documentType"], res["documentNumber"])


"""
  ACHTUNG! ["] not found with text.find, next text is: ``
55о 05`00``
в.д.
Точка №11

  """

if __name__ == '__main__':

  features_dicts = []
  count = 0

  for c in read_all_contracts():
    # doctype = c['documentType']
    contract = join_paragraphs(c, c['_id'])

    _doc_features = doc_line_features(contract)
    features_dicts += _doc_features

    count += 1
    if count > MAX_DOCS: break

  featuresX_data = pd.DataFrame.from_records(features_dicts)

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

  rf = RandomForestRegressor(n_estimators=150, random_state=42, min_samples_split=8)
  # Train the model
  rf.fit(train_features, train_labels)

  # Use the forest's predict method on the test data
  predictions = rf.predict(test_features)
  # # Calculate the absolute errors
  errors = abs(predictions - test_labels)
  # # Print out the mean absolute error (mae)
  print('Mean Absolute Error:', round(np.mean(errors), 4), 'degrees.')

  dump(rf, os.path.join(models_path, 'rf_headers_detector_model.joblib'))
