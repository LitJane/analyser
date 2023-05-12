#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
import random
import warnings

import numpy as np

from analyser.documents import CaseNormalizer
from analyser.structures import OrgStructuralLevel, ContractSubject
from analyser.text_tools import dist_mean_cosine, min_index, Tokens
from analyser.transaction_values import ValueConstraint

# DIST_FUNC = dist_frechet_cosine_undirected
DIST_FUNC = dist_mean_cosine
# DIST_FUNC = dist_cosine_housedorff_undirected
PATTERN_THRESHOLD = 0.75  # 0...1


class FuzzyPattern():

  def __init__(self, prefix_pattern_suffix_tuple, _name='undefined'):
    # assert prefix_pattern_suffix_tuple is not None
    # assert prefix_pattern_suffix_tuple[0].strip() == prefix_pattern_suffix_tuple[0], f'{_name}: {prefix_pattern_suffix_tuple} '
    # assert prefix_pattern_suffix_tuple[2].strip() == prefix_pattern_suffix_tuple[2], f'{_name}: {prefix_pattern_suffix_tuple} '

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
    assert self.embeddings is not None
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
    assert self.embeddings is not None
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

  def calc_exclusive_distances(self, text_ebd) -> ([float], [], {}):
    warnings.warn("calc_exclusive_distances is deprecated ", DeprecationWarning)
    distances_per_pattern = np.zeros((len(self.patterns), len(text_ebd)))

    for pattern_index in range(len(self.patterns)):
      pattern = self.patterns[pattern_index]
      distances_sum = pattern._find_patterns(text_ebd)
      distances_per_pattern[pattern_index] = distances_sum

    # invert
    distances_per_pattern *= -1
    distances_per_pattern = self.onehot_column(distances_per_pattern, None)
    distances_per_pattern *= -1

    # p1 [ [ min, max, mean  ] [ d1, d2, d3, nan, d5 ... ] ]
    # p2 [ [ min, max, mean  ] [ d1, d2, d3, nan, d5 ... ] ]
    ranges: [[float, float, float]] = []
    for row in distances_per_pattern:
      b = row

      if len(b) > 0:
        min = np.nanmin(b)
        max = np.nanmax(b)
        mean = np.nanmean(b)
        ranges.append([min, max, mean])
      else:
        _id = len(ranges)
        print("WARNING: never winning pattern detected! index:", _id, self.patterns[_id])
        ranges.append([np.inf, -np.inf, 0])

    winning_patterns = {}
    for row_index in range(len(distances_per_pattern)):
      row = distances_per_pattern[row_index]
      for col_i in range(len(row)):
        if not np.isnan(row[col_i]):
          winning_patterns[col_i] = (row_index, row[col_i])

    return distances_per_pattern, ranges, winning_patterns


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
    # distances_per_pattern_dict[pat.name] = dists
    dists.flags.writeable = False

  except Exception as e:
    print('ERROR: calculate_distances_per_pattern ', e)
    dists = np.zeros(len(embeddings))
  return dists


def make_smart_meta_click_pattern(attention_vector, embeddings, name=None):
  if attention_vector is None:
    raise ValueError("please provide non empty attention_vector")

  if name is None:
    name = 's-meta-na-' + str(random.random())

  best_id = np.argmax(attention_vector)
  confidence = attention_vector[best_id]
  best_embedding_v = embeddings[best_id]
  meta_pattern = FuzzyPattern(('', ' ', ''), _name=name)
  meta_pattern.embeddings = np.array([best_embedding_v])

  return meta_pattern, confidence, best_id


""" 💔🛐  ===========================📈=================================  ✂️ """

AV_SOFT = 'soft$.'
AV_PREFIX = '$at_'


class PatternMatch():

  def __init__(self, region):
    warnings.warn("use SemanticTag", DeprecationWarning)

    self.subject_mapping = {
      'subj': ContractSubject.Other,
      'confidence': 0
    }
    self.constraints: [ValueConstraint] = []
    self.region: slice = region
    self.confidence: float = 0
    self.pattern_prefix: str = None
    self.attention_vector_name: str = None
    self.parent = None  # 'LegalDocument'

  def get_attention(self, name=None):
    warnings.warn("use SemanticTag", DeprecationWarning)
    if name is None:
      return self.parent.distances_per_pattern_dict[self.attention_vector_name][self.region]
    else:
      return self.parent.distances_per_pattern_dict[name][self.region]

  def get_index(self):
    warnings.warn("use SemanticTag", DeprecationWarning)
    return self.region.start

  key_index = property(get_index)

  def get_tokens(self):
    return self.parent.tokens[self.region]

  tokens = property(get_tokens)


class PatternSearchResult(PatternMatch):
  def __init__(self, org_level: OrgStructuralLevel, region):
    warnings.warn("use SemanticTag", DeprecationWarning)
    super().__init__(region)
    self.org_level: OrgStructuralLevel = org_level


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
