#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
import random

from documents import CaseNormalizer
from ml_tools import relu, filter_values_by_key_prefix, rectifyed_sum
from structures import ContractSubject, OrgStructuralLevel
from text_tools import *
from transaction_values import ValueConstraint

load_punkt = False
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
    assert pattern_embedding[0][0]
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
    # if window_size > len(_text):
    #   print('---ERROR: pattern: "{}" window:{} > len(_text):{} (padding={} mult={})'.format(self.name, window_size, len(_text), whd_padding, wnd_mult)  )
    #   return None

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

      if len(b):
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


class CoumpoundFuzzyPattern(CompoundPattern):
  """
  finds average
  """

  def __init__(self, name="no name"):
    self.name = name
    self.patterns = {}

  def add_pattern(self, pat, weight=1.0):
    assert pat is not None
    self.patterns[pat] = weight

  def find(self, text_ebd):
    sums = self._find_patterns(text_ebd)

    meaninful_sums = sums

    min_i = min_index(meaninful_sums)
    min = sums[min_i]
    mean = meaninful_sums.mean()

    # confidence = sums[min_i] / mean
    sandard_deviation = np.std(meaninful_sums)
    deviation_from_mean = abs(min - mean)
    confidence = sandard_deviation / deviation_from_mean
    return min_i, sums, confidence

  def _find_patterns(self, text_ebd):
    sums = np.zeros(len(text_ebd))
    total_weight = 0
    for p in self.patterns:
      # print('CoumpoundFuzzyPattern, finding', str(p))
      weight = self.patterns[p]
      sp = p._find_patterns(text_ebd)

      sums += sp * weight
      total_weight += abs(weight)
    # norm
    sums /= total_weight
    return sums


class AbstractPatternFactory:

  def __init__(self, embedder):
    self.embedder = embedder  # TODO: do not keep it here, take as an argument for embedd()
    self.patterns: List[FuzzyPattern] = []
    self.patterns_dict = {}

  def create_pattern(self, pattern_name, prefix_pattern_suffix_tuples):
    fp = FuzzyPattern(prefix_pattern_suffix_tuples, pattern_name)
    self.patterns.append(fp)
    self.patterns_dict[pattern_name] = fp
    return fp

  def embedd(self):
    # collect patterns texts
    arr = []
    for p in self.patterns:
      arr.append(p.prefix_pattern_suffix_tuple)

    # =========
    patterns_emb, regions = self.embedder.embedd_contextualized_patterns(arr)
    assert len(patterns_emb) == len(self.patterns)
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

    assert cnt > 0

    av_emb /= cnt

    return np.reshape(av_emb, (1, embedding_vector_len))

  def make_average_pattern(self, pattern_prefix):
    emb = self.average_embedding_pattern(pattern_prefix)

    pat = FuzzyPattern((), pattern_prefix)
    pat.embeddings = emb

    return pat


_case_normalizer = CaseNormalizer()


class AbstractPatternFactoryLowCase(AbstractPatternFactory):
  def __init__(self, embedder):
    AbstractPatternFactory.__init__(self, embedder)
    self.patterns_dict = {}

  def create_pattern(self, pattern_name, ppp):
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
  assert attention_vector is not None
  if name is None:
    name = 's-meta-na-' + str(random.random())

  best_id = np.argmax(attention_vector)
  confidence = attention_vector[best_id]
  best_embedding_v = embeddings[best_id]
  meta_pattern = FuzzyPattern(('', ' ', ''), _name=name)
  meta_pattern.embeddings = np.array([best_embedding_v])

  return meta_pattern, confidence, best_id


""" 💔🛐  ===========================📈=================================  ✂️ """


def improve_attention_vector(embeddings, vv, relu_th=0.5, mix=1):
  assert vv is not None
  meta_pattern, meta_pattern_confidence, best_id = make_smart_meta_click_pattern(vv, embeddings)
  meta_pattern_attention_v = make_pattern_attention_vector(meta_pattern, embeddings)
  meta_pattern_attention_v = relu(meta_pattern_attention_v, relu_th)

  meta_pattern_attention_v = meta_pattern_attention_v * mix + vv * (1.0 - mix)
  return meta_pattern_attention_v, best_id


""" ❤️  =============================📈=================================  ✂️ """


def make_improved_attention_vector(distances_per_pattern_dict, embeddings, pattern_prefix, relu_th: float):
  vvvvv = filter_values_by_key_prefix(distances_per_pattern_dict, pattern_prefix)
  _max_hit_attention, _ = rectifyed_sum(vvvvv, relu_th)
  improved = improve_attention_vector(embeddings, _max_hit_attention, mix=1)
  return improved


AV_SOFT = 'soft$.'
AV_PREFIX = '$at_'


class PatternMatch():

  def __init__(self, region):
    warnings.warn("use SemanticTag", DeprecationWarning)
    assert region.stop - region.start > 0
    self.subject_mapping = {
      'subj': ContractSubject.Other,
      'confidence': 0
    }
    self.constraints: List[ValueConstraint] = []
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
    super(PatternSearchResult, self).__init__(region)

    self.org_level: OrgStructuralLevel = org_level


class ConstraintsSearchResult:
  def __init__(self):
    warnings.warn("ConstraintsSearchResult is deprecated, use PatternSearchResult.constraints", DeprecationWarning)
    self.constraints: List[ValueConstraint] = []
    self.subdoc: PatternSearchResult = None

  def get_context(self) -> PatternSearchResult:  # alias
    warnings.warn("ConstraintsSearchResult is deprecated, use PatternSearchResult.constraints", DeprecationWarning)
    return self.subdoc

  context = property(get_context)


PatternSearchResults = List[PatternSearchResult]
