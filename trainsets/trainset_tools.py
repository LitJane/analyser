import pickle
import random
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from analyser.legal_docs import LegalDocument
from analyser.structures import ContractSubject

VALIDATION_SET_PROPORTION = 0.25


def get_feature_log_weights(trainset_rows, category_column_name):
  subj_count = trainset_rows[category_column_name].value_counts()

  subjects_weights = 1. / np.log(1. + subj_count)

  subjects_weights /= subjects_weights.sum()
  subjects_weights *= len(subjects_weights)

  return subjects_weights


class TrainsetBalancer:

  def __init__(self):
    pass

  def get_indices_split(self, df: DataFrame, category_column_name: str, test_proportion=VALIDATION_SET_PROPORTION) -> (
          [int], [int]):
    random.seed(42)
    cat_count = df[category_column_name].value_counts()  # distribution by category

    _bags = {key: [] for key in cat_count.index}

    _idx: int = 0
    for _, row in df.iterrows():
      subj_code = row[category_column_name]
      _bags[subj_code].append(_idx)
      _idx += 1

    _desired_number_of_samples = max(cat_count.values)
    for subj_code in _bags:
      bag = _bags[subj_code]
      if len(bag) < _desired_number_of_samples:
        repeats = int(_desired_number_of_samples / len(bag))
        bag = sorted(np.tile(bag, repeats))
        _bags[subj_code] = bag

    train_indices = []
    test_indices = []

    for subj_code in _bags:
      bag = _bags[subj_code]
      split_index: int = int(len(bag) * test_proportion)

      train_indices += bag[split_index:]
      test_indices += bag[:split_index]

    # remove instesection
    intersection = np.intersect1d(test_indices, train_indices)
    test_indices = [e for e in test_indices if e not in intersection]

    # shuffle
    random.shuffle(test_indices)
    random.shuffle(train_indices)

    return train_indices, test_indices


class SubjectTrainsetManager:
  EMB_NOISE_AMOUNT = 0.05
  OUTLIERS_PERCENT = 0.05
  NOISY_SAMPLES_AMOUNT = 0.5

  def __init__(self, trainset_description_csv: str):
    self.outliers_percent = SubjectTrainsetManager.OUTLIERS_PERCENT

    '''
    percentange of samples to add noise
    '''
    self.noisy_samples_amount = SubjectTrainsetManager.NOISY_SAMPLES_AMOUNT

    '''
    amount of noise
    '''
    self.noise_amount = SubjectTrainsetManager.EMB_NOISE_AMOUNT

    self.embeddings_cache = {}

    self.csv_path = trainset_description_csv

    # -------
    self.trainset_rows: DataFrame = self._read_trainset_meta()
    self.remove_duplicate_docs()

    self.train_indices, self.test_indices = self.balance_trainset(self.trainset_rows)
    self.subject_name_1hot_map = self._encode_1_hot()

    self.number_of_classes = len(list(self.subject_name_1hot_map.values())[0])

  def print_parameters(self):
    print(f'outliers_percent={self.outliers_percent}')
    print(f'noisy_samples_amount={self.noisy_samples_amount}')
    print(f'noise_amount={self.noise_amount}')

  @staticmethod
  def balance_trainset(trainset_dataframe: DataFrame):
    tb = TrainsetBalancer()
    return tb.get_indices_split(trainset_dataframe, 'subject')

  def _read_trainset_meta(self) -> DataFrame:
    _trainset_meta = pd.read_csv(self.csv_path, encoding='utf-8')
    self._trainset_meta = _trainset_meta
    trainset_rows: DataFrame = _trainset_meta[_trainset_meta['valid']]

    self.subj_count = trainset_rows['subject'].value_counts()

    print(trainset_rows.info())
    return trainset_rows

  @staticmethod
  def _encode_1_hot():
    return ContractSubject.encode_1_hot()

  def noise_embedding(self, emb, var=0.1):
    warnings.warn('must be noised by keras model', DeprecationWarning)
    _mean = 0
    sigma = var ** 0.5
    gauss = np.random.normal(_mean, sigma, emb.shape)
    return emb + gauss

  def _noise_amount(self, subj):
    warnings.warn('must be noised by keras model', DeprecationWarning)
    subj_popularity = self.subj_count[subj]
    max_pop = max(self.subj_count.values)
    return 1 - subj_popularity / max_pop

  def get_embeddings_raw(self, _filename):
    # TODO:::
    filename = _filename

    if filename not in self.embeddings_cache:
      with open(filename, "rb") as pickle_in:
        self.load = pickle.load  # TODO: wtf?
        doc: LegalDocument = self.load(pickle_in)
        self.embeddings_cache[filename] = doc.embeddings
        # todo: do not overload!!

    return self.embeddings_cache[filename]

  def make_fake_outlier(self, emb):
    warnings.warn('use pre-selected real doc outlies', DeprecationWarning)
    label = self.subject_name_1hot_map['Other']
    _mode = np.random.choice([1, 2, 3, 4])

    if _mode == 1:
      return emb * -1, label

    elif _mode == 2:
      return self.noise_embedding(emb, 3), label

    elif _mode == 3:
      return emb * -0.5, label

    elif _mode == 4:
      return np.zeros_like(emb), label

  def remove_duplicate_docs(self, len_threshold=5):
    print(f'sorting by len, {len(self.trainset_rows)}')
    sorted_by_len = self.trainset_rows.sort_values(['confidence', 'len'])

    duplicates = []
    last_row = sorted_by_len.iloc[0]
    for i in range(1, len(sorted_by_len)):
      row = sorted_by_len.iloc[i]
      deltalen = abs(row.len - last_row.len)
      if deltalen < len_threshold:
        duplicates.append(row._id)

      last_row = row

    cleaned_up = self.trainset_rows[~self.trainset_rows['_id'].isin(duplicates)]
    cleaned_up = cleaned_up.sort_values(['_id'])
    self.trainset_rows = cleaned_up

    print(f'new len, {len(self.trainset_rows)}')
    return duplicates
