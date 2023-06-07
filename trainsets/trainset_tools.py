import random

import numpy as np
import pandas as pd
from pandas import DataFrame

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

