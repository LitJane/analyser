#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

import json
import logging
import os
import pathlib
import random
import warnings
from datetime import datetime
from functools import lru_cache
from math import log1p

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bson import json_util
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from packaging import version
from pandas import DataFrame
from pymongo import ASCENDING
from sklearn.metrics import classification_report

from analyser.documents import TextMap
from analyser.finalizer import get_doc_by_id
from analyser.headers_detector import get_tokens_features
from analyser.hyperparams import models_path
from analyser.hyperparams import work_dir as default_work_dir
from analyser.legal_docs import embedd_tokens
from analyser.persistence import DbJsonDoc
from analyser.structures import ContractSubject
from colab_support.renderer import plot_cm
from integration.db import get_mongodb_connection
from tf_support import super_contract_model
from tf_support.embedder_elmo import ElmoEmbedder
from tf_support.super_contract_model import uber_detection_model_005_1_1, seq_labels_contract, \
  seq_labels_contract_swap_orgs
from tf_support.tools import KerasTrainingContext
from trainsets.trainset_tools import split_trainset_evenly, get_feature_log_weights

matplotlib.use('Agg')
logger = logging.getLogger('retrain_contract_uber_model')
logger.setLevel(logging.DEBUG)

SAVE_PICKLES = False
_DEV_MODE = False
_EMBEDD = True


# TODO: 2. use averaged tags confidence for sample weighting
# TODO: 3. evaluate on user-marked documents only


def pad_things(xx, maxlen, padding='post'):
  for x in xx:
    _v = x.mean()
    yield pad_sequences([x], maxlen=maxlen, padding=padding, truncating=padding, value=_v, dtype='float32')[0]


def _get_semantic_map(doc: DbJsonDoc, confidence_override=None) -> DataFrame:
  _len = len(doc)

  df = DataFrame()
  attributes = doc.get_attributes_OLD()
  for _kind, tag in attributes.items():

    _span = tag['span']
    _conf = tag['confidence']
    if confidence_override is not None:
      _conf = confidence_override

    av = np.zeros(_len)  # attention_vector
    av[_span[0]:_span[1]] = _conf #TODO: confidence might be too low, please check this is safe
    df[_kind] = av

  # add missing columns
  for sl in seq_labels_contract:
    if sl not in df:
      df[sl] = np.zeros(_len)

  order = seq_labels_contract
  s1 = doc.get_attr_span_start('org-1-name')
  s2 = doc.get_attr_span_start('org-2-name')
  if s1 is None or (s2 is not None and s2 < s1):
    order = seq_labels_contract_swap_orgs

  av = np.zeros(_len)  # attention_vector
  headers = doc.analysis['headers']
  for h in headers:
    av[h['span'][0]:h['span'][1]] = 1.0

  df['headline_h1'] = av  # make_headline_attention_vector(doc)  ##adding headers

  return df[order]  # re-order columns


class UberModelTrainsetManager:

  def __init__(self, work_dir: str, model_variant_fn=uber_detection_model_005_1_1):
    self.model_variant_fn = model_variant_fn
    self.work_dir: str = work_dir
    self.reports_dir = os.path.join(self.work_dir, 'reports')
    pathlib.Path(self.reports_dir).mkdir(parents=True, exist_ok=True)

    self.stats: DataFrame = self.load_contract_trainset_meta()

  def save_contract_data_arrays(self, db_json_doc: DbJsonDoc, id_override=None):
    # TODO: trim long documens according to contract parser

    id_ = db_json_doc.get_id()
    if id_override is not None:
      id_ = id_override

    embedder = ElmoEmbedder.get_instance('elmo')  # lazy init

    tokens_map: TextMap = db_json_doc.get_tokens_for_embedding()

    # 1) EMBEDDINGS
    embeddings = embedd_tokens(tokens_map,
                               embedder,
                               log_key=f'id={id_} chs={tokens_map.get_checksum()}')

    # 2) TOKEN FEATURES
    token_features: DataFrame = get_tokens_features(db_json_doc.get_tokens_map_unchaged().tokens)

    # 3) SEMANTIC MAP
    semantic_map: DataFrame = _get_semantic_map(db_json_doc, 1.0)
    #####
    if embeddings.shape[0] != token_features.shape[0]:
      msg = f'{id_} embeddings.shape {embeddings.shape} is incompatible with token_features.shape {token_features.shape}'
      raise AssertionError(msg)

    if embeddings.shape[0] != semantic_map.shape[0]:
      msg = f'{id_} embeddings.shape {embeddings.shape} is incompatible with semantic_map.shape {semantic_map.shape}'
      raise AssertionError(msg)

    np.save(self._dp_fn(id_, 'token_features'), token_features)
    np.save(self._dp_fn(id_, 'semantic_map'), semantic_map)
    np.save(self._dp_fn(id_, 'embeddings'), embeddings)

  def save_contract_datapoint(self, d: DbJsonDoc):
    _id = str(d.get_id())
    stats = self.stats  # shortcut

    try:
      self.save_contract_data_arrays(d)

      stats.at[_id, 'checksum'] = d.get_tokens_for_embedding().get_checksum()
      stats.at[_id, 'version'] = d.analysis['version']

      stats.at[_id, 'export_date'] = datetime.now()
      stats.at[_id, 'analyze_date'] = d.analysis['analyze_timestamp']

      subj_att = d.get_subject()
      stats.at[_id, 'subject'] = subj_att['value']
      _value = d.get_attribute('sign_value_currency/value')['value']
      stats.at[_id, 'value'] = _value
      if _value is not None:
        stats.at[_id, 'value_log1p'] = log1p(_value)
      stats.at[_id, 'org-1-alias'] = d.get_attribute('org-1-alias')['value']
      stats.at[_id, 'org-2-alias'] = d.get_attribute('org-2-alias')['value']
      stats.at[_id, 'value_span'] = d.get_attribute('sign_value_currency/value')['span'][0]

      stats.at[_id, 'subject confidence'] = subj_att['confidence']

      if d.user is not None:
        stats.at[_id, 'user_correction_date'] = d.user['updateDate']
    except KeyError as e:
      logger.error(e)
      stats.at[_id, 'valid'] = False

  def get_updated_contracts(self):
    self.lastdate = datetime(1900, 1, 1)
    if len(self.stats) > 0:
      # self.stats.sort_values(["user_correction_date", 'analyze_date', 'export_date'], inplace=True, ascending=False)
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
    # sorting = None
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

  def import_recent_contracts(self):
    self.stats: DataFrame = self.load_contract_trainset_meta()

    docs_ids = [i["_id"] for i in self.get_updated_contracts()]

    for oid in docs_ids:
      d = get_doc_by_id(oid)
      self.save_contract_datapoint(DbJsonDoc(d))
      self._save_stats()

    # export_docs_to_single_json(docs, self.work_dir)

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
    ctx = KerasTrainingContext(self.work_dir)

    model_name = self.model_variant_fn.__name__

    model = self.model_variant_fn(name=model_name, ctx=ctx, trained=True)
    # model.name = model_name

    weights_file_old = os.path.join(models_path, model_name + ".weights")
    weights_file_new = os.path.join(self.work_dir, model_name + ".weights")

    try:
      model.load_weights(weights_file_new, by_name=True, skip_mismatch=True)
      logger.info(f'weights loaded: {weights_file_new}')

    except:
      msg = f'cannot load  {model_name} from  {weights_file_new}'
      warnings.warn(msg)
      model.load_weights(weights_file_old, by_name=True, skip_mismatch=True)
      logger.info(f'weights loaded: {weights_file_old}')

    # freeze bottom 6 layers, including 'embedding_reduced' #TODO: this must be model-specific parameter
    for l in model.layers[0:6]:
      l.trainable = False

    model.compile(loss=super_contract_model.losses, optimizer='Nadam', metrics=super_contract_model.metrics)
    model.summary()

    return model, ctx

  def validate_trainset(self):
    self.stats: DataFrame = self.load_contract_trainset_meta()

    self.stats['valid'] = True
    self.stats['error'] = ''

    for i in self.stats.index:
      try:
        self.make_xyw(i)

      except Exception as e:
        logger.error(e)
        self.stats.at[i, 'valid'] = False
        self.stats.at[i, 'error'] = str(e)

    self._save_stats()

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

  def train(self, generator_factory_method):
    self.stats: DataFrame = self.load_contract_trainset_meta()

    '''
    Phase I: frozen bottom 6 common layers
    Phase 2: all unfrozen, entire trainset, low LR
    :return:
    '''

    batch_size = 24  # TODO: make a param
    train_indices, test_indices = split_trainset_evenly(self.stats, 'subject', seed=66)
    model, ctx = self.init_model()
    ctx.EVALUATE_ONLY = False

    ######################
    ## Phase I retraining
    # frozen bottom layers
    ######################

    ctx.EPOCHS = 30
    ctx.set_batch_size_and_trainset_size(batch_size, len(test_indices), len(train_indices))

    test_gen = generator_factory_method(test_indices, batch_size)
    train_gen = generator_factory_method(train_indices, batch_size, augment_samples=True)

    ctx.train_and_evaluate_model(model, train_gen, test_gen, retrain=True)

    ######################
    ## Phase II finetuning
    #  all unfrozen, entire trainset, low LR
    ######################
    ctx.unfreezeModel(model)
    model.compile(loss=super_contract_model.losses, optimizer='Nadam', metrics=super_contract_model.metrics)
    model.summary()

    ctx.EPOCHS *= 2
    train_gen = generator_factory_method(train_indices + test_indices, batch_size)
    test_gen = generator_factory_method(test_indices, batch_size)
    ctx.train_and_evaluate_model(model, train_gen, test_generator=test_gen, retrain=False, lr=2e-5)

    self.make_training_report(ctx, model)

  def make_training_report(self, ctx: KerasTrainingContext, model: Model):
    ## plot results
    _log = ctx.get_log(model.name)
    if _log is not None:
      _metrics = _log.keys()
      plot_compare_models(ctx, [model.name], _metrics, self.reports_dir)

    _gen = self.make_generator(self.stats.index, 20)
    plot_subject_confusion_matrix(self.reports_dir, model, steps=20, generator=_gen)

  def calculate_samples_weights(self):

    self.stats: DataFrame = self.load_contract_trainset_meta()
    subject_weights = get_feature_log_weights(self.stats, 'subject')

    for i, row in self.stats.iterrows():
      subj_name = row['subject']

      sample_weight = row['subject confidence']
      if not pd.isna(row['user_correction_date']):  # more weight for user-corrected datapoints
        sample_weight = 10.0  # TODO: must be estimated anyhow smartly

      value_weight = 1.0
      if not pd.isna(row['value_log1p']):
        # чтобы всех запутать, вес пропорционален логорифму цены контракта
        # (чтобы было меньше ошибок в контрактах на большие суммы)
        value_weight = row['value_log1p']

      sample_weight *= value_weight
      subject_weight = sample_weight * subject_weights[subj_name]
      self.stats.at[i, 'subject_weight'] = subject_weight
      self.stats.at[i, 'sample_weight'] = sample_weight

    # normalize weights, so the sum == Number of samples
    self.stats.sample_weight /= self.stats.sample_weight.mean()
    self.stats.subject_weight /= self.stats.subject_weight.mean()

    self._save_stats()

  def export_docs_to_json(self):
    self.stats: DataFrame = self.load_contract_trainset_meta()

    docs_ids = [i["_id"] for i in self.get_updated_contracts()]  # Cursor, not list
    export_updated_contracts_to_json(docs_ids, self.work_dir)

  def _dp_fn(self, doc_id, suffix):
    return os.path.join(self.work_dir, 'datasets', f'{doc_id}-datapoint-{suffix}.npy')

  @lru_cache(maxsize=72)
  def make_xyw(self, doc_id):

    row = self.stats.loc[doc_id]

    _subj = row['subject']
    subject_one_hot = ContractSubject.encode_1_hot()[_subj]

    embeddings = np.load(self._dp_fn(doc_id, 'embeddings'))
    token_features = np.load(self._dp_fn(doc_id, 'token_features'))
    semantic_map = np.load(self._dp_fn(doc_id, 'semantic_map'))

    if embeddings.shape[0] != token_features.shape[0]:
      msg = f'{doc_id} embeddings.shape {embeddings.shape} is incompatible with token_features.shape {token_features.shape}'
      raise AssertionError(msg)

    if embeddings.shape[0] != semantic_map.shape[0]:
      msg = f'{doc_id} embeddings.shape {embeddings.shape} is incompatible with semantic_map.shape {semantic_map.shape}'
      raise AssertionError(msg)

    self.stats.at[doc_id, 'error'] = None
    return (
      (embeddings, token_features),
      (semantic_map, subject_one_hot),
      (row['sample_weight'], row['subject_weight']))

  def augment_datapoint(self, dp):
    maxlen = 128 * random.choice([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    cutoff = 16 * random.choice([0, 0, 0, 1, 1, 2, 3])

    return self.trim_maxlen(dp, cutoff, maxlen)

  def trim_maxlen(self, dp, start_from, maxlen):
    (emb, tok_f), (sm, subj), (sample_weight, subject_weight) = dp

    # if emb is not None:  # paranoia, TODO: fail execution, because trainset mut be verifyed in advance

    _padded = [emb, tok_f, sm]

    if start_from > 0:
      _padded = [p[start_from:] for p in _padded]

      # _padded = list(pad_things(_padded, maxlen - start_from, padding='pre'))
    _padded = list(pad_things(_padded, maxlen))

    emb = _padded[0]
    tok_f = _padded[1]
    sm = _padded[2]

    return (emb, tok_f), (sm, subj), (sample_weight, subject_weight)

  def make_generator(self, indices: [int], batch_size: int, augment_samples=False):

    np.random.seed(42)

    while True:
      # next batch
      batch_indices = np.random.choice(a=indices, size=batch_size)

      max_len = 128 * 12
      start_from = 0

      if augment_samples:
        max_len = random.randint(300, 1400)

      batch_input_emb = []
      batch_input_token_f = []
      batch_output_sm = []
      batch_output_subj = []

      weights = []
      weights_subj = []

      # Read in each input, perform preprocessing and get labels
      for doc_id in batch_indices:

        dp = self.make_xyw(doc_id)
        (emb, tok_f), (sm, subj), (sample_weight, subject_weight) = dp

        subject_weight_K = 1.0
        if augment_samples:
          start_from = 0

          row = self.stats.loc[doc_id]
          if random.randint(1, 2) == 1:  # 50% of samples
            segment_center = random.randint(0, len(emb) - 1)  ##select random token as a center
            if not pd.isna(row['value_span']) and random.random() < 0.7:
              segment_center = int(row['value_span'])

            _off = random.randint(max_len // 4, max_len // 2)
            start_from = segment_center - _off
            if start_from < 0:
              start_from = 0
            subject_weight_K = 0.1  # lower subject weight because there mighе be no information about subject around doc. value

        dp = self.trim_maxlen(dp, start_from, max_len)
        # TODO: find samples maxlen

        (emb, tok_f), (sm, subj), (sample_weight, subject_weight) = dp
        subject_weight *= subject_weight_K

        batch_input_emb.append(emb)
        batch_input_token_f.append(tok_f)

        batch_output_sm.append(sm)
        batch_output_subj.append(subj)

        weights.append(sample_weight)
        weights_subj.append(subject_weight)
        # end if emb
      # end for loop

      # Return a tuple of (input, output, weights) to feed the network
      yield ([np.array(batch_input_emb), np.array(batch_input_token_f)],
             [np.array(batch_output_sm), np.array(batch_output_subj)],
             [np.array(weights), np.array(weights_subj)])

  def prepare_trainst(self):
    '''
    1. importing fresh docs
    2. make train samples
    3. validate & clean-up
    :return:
    '''
    self.import_recent_contracts()
    self.calculate_samples_weights()
    self.validate_trainset()
    self.describe_trainset()

  def run(self):
    self.prepare_trainst()

    self.train(self.make_generator)


def export_updated_contracts_to_json(document_ids, work_dir):
  arr = {}
  n = 0
  for k, doc_id in enumerate(document_ids):
    d = get_doc_by_id(doc_id)
    # if '_id' not in d['user']['author']:
    #   print(f'error: user attributes doc {d["_id"]} is not linked to any user')

    if 'auditId' not in d:
      logger.warning(f'error: doc {d["_id"]} is not linked to any audit')

    arr[str(d['_id'])] = d
    # arr.append(d)
    logger.debug(f"exporting JSON {k} {d['_id']}")
    n = k

  with open(os.path.join(work_dir, 'contracts_mongo.json'), 'w', encoding='utf-8') as outfile:
    json.dump(arr, outfile, indent=2, ensure_ascii=False, default=json_util.default)

  logger.info(f'EXPORTED {n} docs')


def onehots2labels(preds):
  _x = np.argmax(preds, axis=-1)
  return [ContractSubject(k).name for k in _x]


def plot_subject_confusion_matrix(reports_path, model, steps=12, generator=None):
  all_predictions = []
  all_originals = []

  for _ in range(steps):
    x, y, _ = next(generator)

    orig_test_labels = onehots2labels(y[1])

    _preds = onehots2labels(model.predict(x)[1])
    # _labels = sorted(np.unique(orig_test_labels + _preds))

    all_predictions += _preds
    all_originals += orig_test_labels

  plot_cm(all_originals, all_predictions)

  img_path = os.path.join(reports_path, f'subjects-confusion-matrix-{model.name}.png')
  plt.savefig(img_path, bbox_inches='tight')

  report = classification_report(all_originals, all_predictions, digits=3)

  print(report)
  with open(os.path.join(reports_path, f'subjects-classification_report-{model.name}.txt'), "w") as text_file:
    text_file.write(report)


def plot_compare_models(ctx, models: [str], metrics, image_save_path):
  _metrics = [m for m in metrics if not m.startswith('val_')]

  for _, m in enumerate(models):

    data: pd.DataFrame = ctx.get_log(m)

    if data is not None:
      data.set_index('epoch')

      for metric in _metrics:
        plt.figure(figsize=(16, 6))
        plt.grid()
        plt.title(f'{metric}')
        for metric_variant in ['', 'val_']:
          key = metric_variant + metric
          if key in data:

            x = data['epoch'][-100:]
            y = data[key][-100:]

            c = 'red'  # plt.cm.jet_r(i * colorstep)
            if metric_variant == '':
              c = 'blue'
            plt.plot(x, y, label=f'{key}', alpha=0.2, color=c)

            y = y.rolling(4, win_type='gaussian').mean(std=4)
            plt.plot(x, y, label=f'{key} SMOOTH', color=c)

            plt.legend(loc='upper right')

        img_path = os.path.join(image_save_path, f'{m}-{metric}.png')
        plt.savefig(img_path, bbox_inches='tight')

    else:
      logger.error('cannot plot')


if __name__ == '__main__':
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  logger.addHandler(ch)

  '''
  0. Read 'contract_trainset_meta.csv CSV, find the last datetime of export
  1. Fetch recent docs from DB: update date > last datetime of export 
  2. Embedd them, save embeddings, save other features
  
  '''

  # os.environ['GPN_DB_NAME'] = 'gpn'
  # os.environ['GPN_DB_HOST'] = '192.168.10.36'
  # os.environ['GPN_DB_PORT'] = '27017'
  # db = get_mongodb_connection()
  #

  umtm = UberModelTrainsetManager(default_work_dir)
  umtm.run()

  # umtm.import_recent_contracts()
  # umtm.calculate_samples_weights()
  #
  # model, ctx = umtm.init_model()
  # umtm.make_training_report(ctx, model)
