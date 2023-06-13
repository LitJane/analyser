#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
import warnings

import numpy as np

from analyser.documents import CaseNormalizer
from analyser.text_tools import dist_mean_cosine, min_index, Tokens
from analyser.transaction_values import ValueConstraint

DIST_FUNC = dist_mean_cosine

PATTERN_THRESHOLD = 0.75  # 0...1


class FuzzyPattern():

  def __init__(self, prefix_pattern_suffix_tuple, _name='undefined'):

    self.prefix_pattern_suffix_tuple = prefix_pattern_suffix_tuple
    self.name = _name
    self.soft_sliding_window_borders = False
    self.embeddings = None
    self.region = None

  def set_embeddings(self, pattern_embedding, region=None):
    # TODO: check dimensions

    self.embeddings = pattern_embedding
    self.region = region

  def _eval_distances(self, _text, dist_function=DIST_FUNC, whd_padding=0, wnd_mult=1):
    if self.embeddings is None:
      raise ValueError

    """
      For each token in the given sentences, it calculates the semantic distance to
      each and every pattern in _pattens arg.

      WARNING: may return None!

      TODO: tune sliding window size
    """

    _distances = np.ones(len(_text))

    _pat = self.embeddings

    window_size = wnd_mult * len(_pat) + whd_padding

    for word_index in range(0, len(_text)):
      _fragment = _text[word_index: word_index + window_size]
      _distances[word_index] = dist_function(_fragment, _pat)

    return _distances

  def _eval_distances_multi_window(self, _text, dist_function=DIST_FUNC):

    if self.embeddings is None:
      raise ValueError
    distances = [self._eval_distances(_text, dist_function, whd_padding=0, wnd_mult=1)]

    if self.soft_sliding_window_borders:
      distances.append(self._eval_distances(_text, dist_function, whd_padding=2, wnd_mult=1))
      distances.append(self._eval_distances(_text, dist_function, whd_padding=1, wnd_mult=2))
      distances.append(self._eval_distances(_text, dist_function, whd_padding=7, wnd_mult=0))

    sum = None
    cnt = 0
    for d in distances:
      if d is not None:
        cnt = cnt + 1
        if sum is None:
          sum = np.array(d)
        else:
          sum += d

    assert cnt > 0
    sum = sum / cnt

    return sum

  def _find_patterns(self, text_ebd):
    """
      text_ebd:  tensor of embeedings
    """
    distances = self._eval_distances_multi_window(text_ebd)
    return distances

  def find(self, text_ebd):
    """
      text_ebd:  tensor of embeedings
    """

    sums = self._find_patterns(text_ebd)
    min_i = min_index(sums)  # index of the word with minimum distance to the pattern

    return min_i, sums

  def __str__(self):
    return ' '.join(['FuzzyPattern:', str(self.name), str(self.prefix_pattern_suffix_tuple)])


class CompoundPattern:
  def __init__(self):
    pass


class ExclusivePattern(CompoundPattern):

  def __init__(self):
    self.patterns = []

  def add_pattern(self, pat):
    self.patterns.append(pat)

  def onehot_column(self, a, mask=-2 ** 32):
    """

    keeps only maximum in every column. Other elements are replaced with mask

    :param a:
    :param mask:
    :return:
    """
    maximals = np.max(a, 0)

    for i in range(a.shape[0]):
      for j in range(a.shape[1]):
        if a[i, j] < maximals[j]:
          a[i, j] = mask

    return a


class AbstractPatternFactory:

  def __init__(self):
    self.patterns: [FuzzyPattern] = []
    self.patterns_dict = {}

  def create_pattern(self, pattern_name, prefix_pattern_suffix_tuples):
    fp = FuzzyPattern(prefix_pattern_suffix_tuples, pattern_name)
    self.patterns.append(fp)
    self.patterns_dict[pattern_name] = fp
    return fp

  def embedd(self, embedder):
    # collect patterns texts
    arr = []
    for p in self.patterns:
      arr.append(p.prefix_pattern_suffix_tuple)

    # =========
    patterns_emb, regions = embedder.embedd_contextualized_patterns(arr)
    if len(patterns_emb) != len(self.patterns):
      raise RuntimeError("len(patterns_emb) != len(self.patterns)")
    # =========

    for i in range(len(patterns_emb)):
      self.patterns[i].set_embeddings(patterns_emb[i], regions[i])

  def average_embedding_pattern(self, pattern_prefix):
    av_emb = None
    cnt = 0
    embedding_vector_len = None
    for p in self.patterns:

      if p.name[0: len(pattern_prefix)] == pattern_prefix:
        embedding_vector_len = p.embeddings.shape[1]
        cnt += 1
        p_av_emb = np.mean(p.embeddings, axis=0)
        if av_emb is None:
          av_emb = np.array(p_av_emb)
        else:
          av_emb += p_av_emb

    if cnt <= 0:
      raise RuntimeError("count must be >0")

    av_emb /= cnt

    return np.reshape(av_emb, (1, embedding_vector_len))

  def make_average_pattern(self, pattern_prefix):
    emb = self.average_embedding_pattern(pattern_prefix)

    pat = FuzzyPattern((), pattern_prefix)
    pat.embeddings = emb

    return pat


_case_normalizer = CaseNormalizer()


class AbstractPatternFactoryLowCase(AbstractPatternFactory):
  def __init__(self):
    AbstractPatternFactory.__init__(self)
    self.patterns_dict = {}

  def create_pattern(self, pattern_name, ppp: [str]):
    _ppp = (_case_normalizer.normalize_text(ppp[0]),
            _case_normalizer.normalize_text(ppp[1]),
            _case_normalizer.normalize_text(ppp[2]))

    fp = FuzzyPattern(_ppp, _name=pattern_name)

    if pattern_name in self.patterns_dict:
      # Let me be strict!
      e = f'Duplicated {pattern_name}'
      raise ValueError(e)

    self.patterns_dict[pattern_name] = fp
    self.patterns.append(fp)
    return fp


def make_pattern_attention_vector(pat: FuzzyPattern, embeddings, dist_function=DIST_FUNC):
  try:
    dists = pat._eval_distances_multi_window(embeddings, dist_function)

    # TODO: this inversion must be a part of a dist_function
    dists = 1.0 - dists
    dists.flags.writeable = False

  except Exception as e:
    print('ERROR: calculate_distances_per_pattern ', e)
    dists = np.zeros(len(embeddings))
  return dists


class ConstraintsSearchResult:
  def __init__(self):
    warnings.warn("ConstraintsSearchResult is deprecated, use PatternSearchResult.constraints", DeprecationWarning)
    self.constraints: [ValueConstraint] = []
    self.subdoc = None

  def get_context(self):  # alias
    warnings.warn("ConstraintsSearchResult is deprecated, use PatternSearchResult.constraints", DeprecationWarning)
    return self.subdoc

  context = property(get_context)


def create_value_negation_patterns(f: AbstractPatternFactory, name='not_sum_'):
  f.create_pattern(f'{name}1', ('', 'пункт 0.', ''))
  f.create_pattern(f'{name}2', ('', '0 дней', ''))
  f.create_pattern(f'{name}3', ('', 'в течение 0 ( ноля ) дней', ''))
  f.create_pattern(f'{name}4', ('', '0 января', ''))
  f.create_pattern(f'{name}5', ('', '0 минут', ''))
  f.create_pattern(f'{name}6', ('', '0 часов', ''))
  f.create_pattern(f'{name}7', ('', '0 процентов', ''))
  f.create_pattern(f'{name}8', ('', '0 %', ''))
  f.create_pattern(f'{name}9', ('', '0 % голосов', ''))
  f.create_pattern(f'{name}10', ('', '2000 год', ''))
  f.create_pattern(f'{name}11', ('', '0 человек', ''))
  f.create_pattern(f'{name}12', ('', '0 метров', ''))


def create_value_patterns(f: AbstractPatternFactory, name='sum_max_p_'):
  suffix = 'млн. тыс. миллионов тысяч рублей долларов копеек евро'
  _prefix = ''

  f.create_pattern(f'{name}1', (_prefix + 'стоимость', 'не более 0', suffix))
  f.create_pattern(f'{name}2', (_prefix + 'цена', 'не больше 0', suffix))
  f.create_pattern(f'{name}3', (_prefix + 'стоимость <', '0', suffix))
  f.create_pattern(f'{name}4', (_prefix + 'цена менее', '0', suffix))
  f.create_pattern(f'{name}5', (_prefix + 'стоимость не может превышать', '0', suffix))
  f.create_pattern(f'{name}6', (_prefix + 'общая сумма может составить', '0', suffix))
  f.create_pattern(f'{name}7', (_prefix + 'лимит соглашения', '0', suffix))
  f.create_pattern(f'{name}8', (_prefix + 'верхний лимит стоимости', '0', suffix))
  f.create_pattern(f'{name}9', (_prefix + 'максимальная сумма', '0', suffix))


PATTERN_DELIMITER = ':'


def build_sentence_patterns(strings: Tokens, prefix: str, prefix_obj=None):
  ret = []
  for txt in strings:
    ret.append([f'{prefix}{PATTERN_DELIMITER}{len(ret)}', txt, prefix_obj])

  return ret
