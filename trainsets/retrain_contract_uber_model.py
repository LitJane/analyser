#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

import logging
import os
import pathlib
import random
import warnings
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from packaging import version
from pandas import DataFrame
from pymongo import ASCENDING
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from analyser.documents import TextMap
from analyser.headers_detector import get_tokens_features
from analyser.hyperparams import models_path
from analyser.hyperparams import work_dir as default_work_dir
from analyser.legal_docs import embedd_tokens
from analyser.persistence import DbJsonDoc
from analyser.structures import ContractSubject
from integration.db import get_mongodb_connection
from tf_support.embedder_elmo import ElmoEmbedder
from tf_support.super_contract_model import get_semantic_map_new, make_att_model, losses, metrics
from tf_support.tools import KerasTrainingContext

matplotlib.use('Agg')
logger = logging.getLogger('retrain_contract_uber_model')
logger.setLevel(logging.DEBUG)

SAVE_PICKLES = False
_DEV_MODE = False
_EMBEDD = True


# TODO: 2. use averaged tags confidence for sample weighting
# TODO: 3. evaluate on user-marked documents only


def _dp_fn(doc_id, suffix):
  return os.path.join(default_work_dir, 'datasets', f'{doc_id}-datapoint-{suffix}.npy')


def save_contract_data_arrays(db_json_doc: DbJsonDoc):
  embedder = ElmoEmbedder.get_instance('elmo')
  # TODO: trim long documens according to contract parser

  id_ = db_json_doc.get_id()

  tokens_map: TextMap = db_json_doc.get_tokens_for_embedding()

  # 1) EMBEDDINGS
  print(len(tokens_map))
  embeddings = embedd_tokens(tokens_map,
                             embedder,
                             log_key=f'id={id_} chs={tokens_map.get_checksum()}')

  # 2) TOKEN FEATURES
  token_features: DataFrame = get_tokens_features(db_json_doc.get_tokens_map_unchaged().tokens)

  # 3) SEMANTIC MAP
  semantic_map: DataFrame = get_semantic_map_new(db_json_doc)
  #####

  np.save(_dp_fn(id_, 'token_features'), token_features)
  np.save(_dp_fn(id_, 'semantic_map'), semantic_map)
  _embeddings_file = _dp_fn(id_, 'embeddings')
  np.save(_embeddings_file, embeddings)
  print(f'embeddings saved to {_embeddings_file} {embeddings.shape}')


def pad_things(xx, maxlen, padding='post'):
  for x in xx:
    _v = x.mean()
    yield pad_sequences([x], maxlen=maxlen, padding=padding, truncating=padding, value=_v, dtype='float32')[0]


class UberModelTrainsetManager:

  def __init__(self, work_dir: str, reports_dir=None, model_variant_fn=make_att_model):

    self.model_variant_fn = model_variant_fn
    self.work_dir: str = work_dir

    if reports_dir is None:
      self.reports_dir = os.path.join(self.work_dir, 'reports')
    else:
      self.reports_dir = reports_dir

    pathlib.Path(self.work_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(self.reports_dir).mkdir(parents=True, exist_ok=True)

    self.stats: DataFrame = self.load_contract_trainset_meta()

  def get_updated_contracts(self):
    self.lastdate = datetime(1900, 1, 1)
    if len(self.stats) > 0:
      self.lastdate = self.stats[["user_correction_date", 'analyze_date']].max().max()
    logger.info(f'latest export_date: [{self.lastdate}]')

    logger.debug('obtaining DB connection...')
    db = get_mongodb_connection()
    documents_collection = db['documents']

    # TODO: filter by version
    query = {
      '$and': [
        {"parse.documentType": "CONTRACT"},
        {"state": 15},
        {'$or': [
          {"analysis.attributes": {"$ne": None}},
          {"user.attributes": {"$ne": None}}
        ]},

        {'$or': [
          {'analysis.analyze_timestamp': {'$gt': self.lastdate}},
          {'user.updateDate': {'$gt': self.lastdate}}
        ]}
      ]
    }

    logger.debug(f'running DB query {query}')
    # TODO: sorting fails in MONGO
    sorting = [('analysis.analyze_timestamp', ASCENDING),
               ('user.updateDate', ASCENDING)]

    res = documents_collection.find(filter=query, sort=sorting, projection={'_id': True})

    res.limit(600)

    logger.info('running DB query: DONE')

    return res

  @staticmethod
  def _remove_obsolete_datapoints(df: DataFrame):

    if 'valid' not in df:
      df['valid'] = True

    threshold_v = version.parse("1.6.0")
    for i, row in df.iterrows():
      try:
        if pd.isna(row['user_correction_date']):
          if version.parse(row['version']) < threshold_v:
            df.at[i, 'valid'] = False
      except TypeError:
        df.at[i, 'valid'] = False

  def load_contract_trainset_meta(self) -> DataFrame:
    _f = os.path.join(self.work_dir, 'contract_trainset_meta.csv')
    logger.info(f"loading trainset meta from {_f}")
    try:
      df = pd.read_csv(_f, index_col='_id')
      df['user_correction_date'] = pd.to_datetime(df['user_correction_date'])
      df['analyze_date'] = pd.to_datetime(df['analyze_date'])
      df.index.name = '_id'

      logger.info(f'number of samples BEFORE clean-up: {len(df)}')
      df = df[df['valid'] == True]
      df = df[df['subject'] != 'BigDeal']
      logger.info(f'number of samples AFTER clean-up: {len(df)}')

      UberModelTrainsetManager._remove_obsolete_datapoints(df)

      logger.info("OK")
    except FileNotFoundError:
      df = DataFrame(columns=['export_date'])
      df.index.name = '_id'
      logger.info(f"cannot load trainset meta from {_f}, creating blank")

    if 'subject' not in df:
      df['subject'] = 'Other'

    if 'org-1-alias' not in df:
      df['org-1-alias'] = ''

    if 'org-2-alias' not in df:
      df['org-2-alias'] = ''

    df['org-1-alias'] = df['org-1-alias'].fillna('')
    df['org-2-alias'] = df['org-2-alias'].fillna('')
    df['subject'] = df['subject'].fillna('Other')

    logger.info(f'TOTAL DATAPOINTS IN TRAINSET: {len(df)}')
    return df

  def save_stats(self):
    self._save_stats()

  def _save_stats(self):

    # TODO are you sure, you need to drop_duplicates on every step?
    # todo: might be .. move this code to self._save_stats()
    # todo: print trainset stats

    so = []
    if 'user_correction_date' in self.stats:
      so.append('user_correction_date')
    if 'analyze_date' in self.stats:
      so.append('analyze_date')

    if len(so) > 0:
      logger.info(f'docs in meta: {len(self.stats)}')
      self.stats.sort_values(so, inplace=True, ascending=False)
      self.stats.drop_duplicates(subset="checksum", keep='first', inplace=True)
      logger.info(f'docs in meta after drop_duplicates: {len(self.stats)}')

    self.stats.sort_values('value', inplace=True, ascending=False)
    self.stats.to_csv(os.path.join(self.work_dir, 'contract_trainset_meta.csv'), index=True)

  def init_model(self) -> (Model, KerasTrainingContext):
    ctx = KerasTrainingContext(checkpoints_path=self.reports_dir)

    model_name = self.model_variant_fn.__name__

    model = self.model_variant_fn(name=model_name, ctx=ctx, trained=True)

    weights_file_old = os.path.join(models_path, model_name + ".weights")
    weights_file_new = os.path.join(self.reports_dir, model_name + ".weights")

    try:
      model.load_weights(weights_file_new, by_name=True, skip_mismatch=True)
      logger.info(f'weights loaded: {weights_file_new}')

    except Exception as e:
      msg = f'cannot load  {model_name} from  {weights_file_new}: {e}'
      warnings.warn(msg)
      model.load_weights(weights_file_old, by_name=True, skip_mismatch=True)
      logger.info(f'weights loaded: {weights_file_old}')

    # freeze bottom 6 layers, including 'embedding_reduced' #TODO: this must be model-specific parameter
    for layer in model.layers[0:6]:
      layer.trainable = False

    model.compile(loss=losses, optimizer='Nadam', metrics=metrics)

    return model, ctx

  def describe_trainset(self):
    # TODO: report
    self.stats: DataFrame = self.load_contract_trainset_meta()
    subj_count = self.stats['subject'].value_counts()

    # plot subj distribution---------------------
    sns.barplot(subj_count.values, subj_count.index)
    plt.title('Frequency Distribution of subjects')
    plt.xlabel('Number of Occurrences')
    img_path = os.path.join(self.reports_dir, 'contracts-subjects-dist.png')
    plt.savefig(img_path, bbox_inches='tight')

  def _dp_fn(self, doc_id, suffix):
    return os.path.join(self.work_dir, 'datasets', f'{doc_id}-datapoint-{suffix}.npy')

  def augment_datapoint(self, dp):
    maxlen = 128 * random.choice([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    cutoff = 16 * random.choice([0, 0, 0, 1, 1, 2, 3])

    return self.trim_maxlen(dp, cutoff, maxlen)

  @staticmethod
  def trim_maxlen(dp, start_from, maxlen):
    (emb, tok_f), (sm, subj), (sample_weight, subject_weight) = dp

    # if emb is not None:  # paranoia, TODO: fail execution, because trainset mut be verifyed in advance

    _padded = [emb, tok_f, sm]

    if start_from > 0:
      _padded = [p[start_from:] for p in _padded]

    _padded = list(pad_things(_padded, maxlen))

    emb = _padded[0]
    tok_f = _padded[1]
    sm = _padded[2]

    return (emb, tok_f), (sm, subj), (sample_weight, subject_weight)


def onehots2labels(preds):
  _x = np.argmax(preds, axis=-1)
  return [ContractSubject(k).name for k in _x]
