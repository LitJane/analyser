import math
import warnings
from datetime import datetime
from enum import Enum
from typing import List, TypeVar, Iterable, Generic

import numpy as np
import scipy.spatial.distance as distance
from pandas import DataFrame
from scipy import special as scs

from analyser.hyperparams import HyperParameters
from analyser.text_tools import Tokens

Embedding = 'np.ndarray[float]' or [float]
Embeddings = 'np.ndarray[Embedding]' or [Embedding]

FixedVector = TypeVar('FixedVector', List[float], np.ndarray)
Vector = TypeVar('Vector', FixedVector, Iterable[float])
Vectors = TypeVar('Vectors', List[Vector], Iterable[Vector])
FixedVectors = TypeVar('FixedVectors', List[FixedVector], Iterable[FixedVector])

Spans = [(int, int)]

T = TypeVar('T')  # just type variable


def span_to_slice(span) -> slice:
  return slice(span[0], span[1])


class ProbableValue(Generic[T]):
  def __init__(self, value: T, confidence: float):
    self.value: T = value
    self.confidence: float = confidence

  def __str__(self):
    return f'{self.value} \t {self.confidence:2f}'


def split_by_token(tokens: List[str], token):
  res = []
  sentence = []
  for i in tokens:
    if i == token:
      res.append(sentence)
      sentence = []
    else:
      sentence.append(i)

  res.append(sentence)
  return res


def split_by_token_into_ranges(tokens: List, token) -> List[slice]:
  warnings.warn("deprecated", DeprecationWarning)
  res = []

  p = 0
  for i in range(len(tokens)):
    if tokens[i] == token:
      res.append(slice(p, i + 1))
      p = i + 1

  if p != len(tokens):
    res.append(slice(p, len(tokens)))
  return res


def estimate_threshold(a, min_th=0.3):
  return max(min_th, np.max(a) * 0.7)


def normalize(x: FixedVector, out_range=(0, 1)):
  domain = np.min(x), np.max(x)
  if (domain[1] - domain[0]) == 0:
    # all same
    return np.full(len(x), out_range[0])
    # raise ValueError('all elements are same')

  y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
  return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def smooth_safe(x: FixedVector, window_len=10, window='hanning'):
  _blur = int(min([window_len, 2 + len(x) / 3.0]))
  _blur = int(_blur / 2) * 2
  if _blur > (len(x)):
    return x
  return smooth(x, window_len=_blur, window=window)


def smooth(x: FixedVector, window_len=11, window='hanning'):
  """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

  if x.ndim != 1:
    raise ValueError("smooth only accepts 1 dimension arrays.")

  if x.size < window_len:
    raise ValueError("Input vector needs to be bigger than window size.")

  if window_len < 3:
    return x

  if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

  s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]

  if window == 'flat':  # moving average
    w = np.ones(window_len, 'd')
  else:
    w = eval('np.' + window + '(window_len)')

  y = np.convolve(w / w.sum(), s, mode='valid')
  #     return y
  halflen = int(window_len / 2)
  #     return y[0:len(x)]
  return y[(halflen - 1):-halflen]


def relu(x: np.ndarray, relu_th: float = 0.0) -> np.ndarray:
  """deprecated: use np.maximum( )"""
  # assert type(x) is np.ndarray

  _relu = x * (x > relu_th)
  return _relu


def extremums(x: FixedVector) -> List[int]:
  _extremums = [0]
  for i in range(1, len(x) - 1):
    if x[i - 1] < x[i] > x[i + 1]:
      _extremums.append(i)
  return _extremums


def softmax(v: np.ndarray) -> np.ndarray:
  x = normalize(v)
  x /= len(x)
  return x


def make_echo(av: FixedVector, k=0.5) -> np.ndarray:
  innertia = np.zeros(len(av))
  _sum = 0

  for i in range(len(av)):
    if av[i] > k:
      _sum = av[i]
    innertia[i] = _sum
  #     sum-=0.0005
  return innertia


def momentum_(x: FixedVector, decay=0.99) -> np.ndarray:
  innertia = np.zeros(len(x))
  m = 0
  for i, xi in enumerate(x):
    m += xi
    innertia[i] = m
    m *= decay

  return innertia


def momentum(x: FixedVector, decay=0.999) -> np.ndarray:
  innertia = np.zeros(len(x))
  m = 0
  for i, xi in enumerate(x):
    m = max(m, xi)
    innertia[i] = m
    m *= decay

  return innertia


def momentum_t(x: FixedVector, half_decay: int = 10, left=False) -> np.ndarray:
  # assert half_decay > 0

  decay = math.pow(2, -1 / half_decay)
  innertia = np.zeros(len(x))
  m = 0
  _r = range(len(x))
  if left:
    _r = reversed(_r)
  for i in _r:
    m = max(m, x[i])
    innertia[i] = m

    m *= decay

  return innertia


def momentum_p(x, half_decay: int = 10, left=False) -> np.ndarray:
  # assert half_decay > 0

  decay = math.pow(2, -1 / half_decay)
  innertia = np.zeros(len(x))
  m = 0
  _r = range(len(x))
  if left:
    _r = reversed(_r)
  for i in _r:
    m = m + x[i] - (m * x[i])
    innertia[i] = m

    m *= decay

  return innertia


def momentum_reversed(x: FixedVector, decay=0.999) -> np.ndarray:
  innertia = np.zeros(len(x))
  m = 0
  for i in reversed(range(0, len(x))):
    m = max(m, x[i])
    innertia[i] = m
    m *= decay

  return innertia


def onehot_column(a: np.ndarray, mask=-2 ** 32, replacement=None):
  """

  Searches for maximum in every column.
  Other elements are replaced with mask

  :param a:
  :param mask:
  :return:
  """
  maximals = np.max(a, 0)

  for i in range(a.shape[0]):
    for j in range(a.shape[1]):
      if a[i, j] < maximals[j]:
        a[i, j] = mask
      else:
        if replacement is not None:
          a[i, j] = replacement

  return a


def most_popular_in(arr: FixedVector) -> int:
  if len(arr) == 0:
    return np.nan

  counts = np.bincount(arr)
  return int(np.argmax(counts))


def remove_similar_indexes(indexes: List[int], min_section_size=20):
  if len(indexes) < 2:
    return indexes

  indexes_zipped = [indexes[0]]

  for i in range(1, len(indexes)):
    if indexes[i] - indexes[i - 1] > min_section_size:
      indexes_zipped.append(indexes[i])
  return indexes_zipped


def cut_above(x: np.ndarray, threshold: float) -> np.ndarray:
  float___threshold = (x * float(-1.0)) + threshold

  return threshold + relu(float___threshold) * -1.0


def put_if_better(destination: dict, key, x, is_better: staticmethod):
  if key in destination:
    if is_better(x, destination[key]):
      destination[key] = x
  else:
    destination[key] = x


def rectifyed_sum(vectors: FixedVectors, relu_th: float = 0.0) -> np.ndarray:
  _sum = None

  for x in vectors:
    if _sum is None:
      _sum = np.zeros(len(x))
    _sum += relu(x, relu_th)

  # assert _sum is not None

  return _sum


def filter_values_by_key_prefix(dictionary: dict, prefix: str) -> Vectors:
  for p in dictionary:
    if str(p).startswith(prefix):
      yield dictionary[p]


def max_exclusive_pattern_by_prefix(distances_per_pattern_dict, prefix) -> FixedVector:
  vectors = filter_values_by_key_prefix(distances_per_pattern_dict, prefix)

  return max_exclusive_pattern(vectors)


def max_exclusive_pattern(vectors: FixedVectors) -> FixedVector:
  _sum = None
  for x in vectors:
    if _sum is None:
      _sum = np.zeros(len(x))

    _sum = np.maximum(_sum, x)

  return _sum


def conditional_p_sum(vector: FixedVector) -> float:
  _sum = 0
  for x in vector:
    a = _sum + x
    b = _sum * x
    _sum = a - b

  return _sum


def sum_probabilities(vectors: FixedVectors) -> FixedVector:
  _sum = np.zeros_like(vectors[0])
  for x in vectors:
    a = _sum + x
    b = _sum * x
    _sum = a - b

  return _sum


def subtract_probability(a: FixedVector, b: FixedVector) -> FixedVector:
  return 1.0 - sum_probabilities([1.0 - a, b])


class TokensWithAttention:
  def __init__(self, tokens: Tokens, attention: FixedVector):
    warnings.warn("TokensWithAttention is deprecated, use ...", DeprecationWarning)
    # assert len(tokens) == len(attention)
    self.tokens = tokens
    self.attention = attention


TAG_KEY_DELIMITER = '/'


class SemanticTagBase:
  # value: str or Enum or int or float or datetime.date or None = None
  # span: (int, int)

  def __init__(self, tag=None):
    super().__init__()
    self.confidence: float = 0.0
    self.value = None
    if tag is not None:
      self.value = tag.value
      self.set_span(tag.span)
      self.set_confidence(tag.confidence)

  def get_confidence(self) -> float:
    return self.confidence

  def get_span(self) -> (int, int):
    return self.span

  def set_span(self, span: (int, int)):
    self.span = (int(span[0]), int(span[1]))

  def set_confidence(self, confidence: float):
    if confidence is None:
      self.confidence = 0
    else:
      self.confidence = float(confidence)

  def as_json_attribute(self):
    raise NotImplementedError()

  def __len__(self) -> int:
    if self.span:
      return self.span[1] - self.span[0]
    return 0

  def __add__(self, addon: int):
    return self.offset(addon)

  def offset(self, addon: int):
    self.span = self.span[0] + int(addon), self.span[1] + int(addon)
    return self

  def contains(self, child: [int]) -> bool:
    return self.span[0] <= child[0] and child[1] <= self.span[1]

  def as_slice(self):
    return slice(self.span[0], self.span[1])

  slice = property(as_slice)


class SemanticTag(SemanticTagBase):

  def __init__(self,
               kind: str = None,
               value: str or Enum or None = None,
               span: (int, int) = (-1, -1),
               span_map: str or None = 'words',
               parent: 'SemanticTag' = None, confidence: float = 1.0):
    super().__init__()
    self.kind = kind
    self.value: str or Enum or None = value
    '''name of the parent (or group) tag '''

    self._parent_tag: 'SemanticTag' = parent

    if span is not None:
      self.span = (int(span[0]), int(span[1]))  # kind of type checking
    else:
      self.span = (0, 0)  # TODO: might be keep None?
    self.span_map = span_map
    self.confidence = confidence

  def clean_copy(self) -> SemanticTagBase:

    r = SemanticTagBase()
    for a in ['value', 'span', 'confidence']:
      setattr(r, a, getattr(self, a))
    return r

  def as_json_attribute(self):

    key = self.get_key()
    attribute = self.__dict__.copy()

    if isinstance(self.value, Enum):
      attribute['value'] = self.value.name

    del attribute['kind']
    if '_parent_tag' in attribute:
      if self.parent is not None:
        attribute['parent'] = self.parent
      del attribute['_parent_tag']

    return key, attribute

  @staticmethod
  def number_key(base: str or Enum, number: int) -> str:
    if isinstance(base, Enum):
      base = base.name

    return f'{base}-{number}'

  def is_child_of(self, p: 'SemanticTag') -> bool:
    return self._parent_tag == p

  def get_parent(self) -> str or None:
    if self._parent_tag is not None:
      return self._parent_tag.get_key()
    else:
      return None

  parent = property(get_parent)

  def get_key(self):
    key = self.kind.replace('.', '-').replace(TAG_KEY_DELIMITER, '-')
    if self._parent_tag is not None:
      key = TAG_KEY_DELIMITER.join([self._parent_tag.get_key(), key])
    return key

  def isNotEmpty(self) -> bool:
    return self.span is not None and self.span[0] != self.span[1]

  @staticmethod
  def find_by_kind(lst: List['SemanticTag'], kind: str) -> 'SemanticTag':
    for s in lst:
      if s.kind == kind:
        return s

  @staticmethod
  def find_by_kind_and_value(lst: List['SemanticTag'], kind: str, val: str) -> 'SemanticTag':
    for s in lst:
      if s.kind == kind and s.value == val:
        return s

  def set_parent_tag(self, pt):
    self._parent_tag = pt

  def __str__(self):
    return f'SemanticTag: {self.get_key()} {self.span} {self.value} {self.confidence}'


def clean_semantic_tag_copy(t: SemanticTag) -> SemanticTagBase or None:
  if t is None:
    return None

  r = SemanticTagBase()
  for a in ['value', 'span', 'confidence']:
    setattr(r, a, getattr(t, a))
  return r


def clean_semantic_tags_copy(tags: [SemanticTag]) -> [SemanticTagBase]:
  return [clean_semantic_tag_copy(t) for t in tags]


def max_confident_tags(vals: [SemanticTagBase]) -> [SemanticTagBase]:
  if vals:
    return [max(vals, key=lambda a: a.confidence)]
  else:
    return []


def max_confident_tag(vals: [SemanticTagBase]) -> SemanticTagBase:
  if vals:
    return max(vals, key=lambda a: a.confidence)


def estimate_confidence(vector: FixedVector) -> (float, float, int, float):
  if len(vector) == 0:
    return 0, np.nan, 0, np.nan

  sum_ = sum(vector)
  _max = np.max(vector)
  nonzeros_count = len(np.nonzero(vector)[0])
  confidence = 0

  if nonzeros_count > 0:
    confidence = sum_ / nonzeros_count

  return confidence, sum_, nonzeros_count, _max


def estimate_confidence_by_mean_top_non_zeros(x: FixedVector, head_size: int = 10) -> float:
  top10 = sorted(x)[-head_size:]

  nonzeros_count = len(np.nonzero(top10)[0])
  sum_ = sum(top10)
  confidence = 0

  if nonzeros_count > 0:
    confidence = sum_ / nonzeros_count
  return confidence


def estimate_confidence_by_mean_top(x: FixedVector, head_size: int = 10) -> float:
  """
  taking mean of max 10 values
  """
  return float(np.mean(sorted(x)[-head_size:]))


def combined_attention_vectors(vectors_dict, vector_names):
  vectors = [vectors_dict[v] for v in vector_names]
  return sum_probabilities(vectors)


sum_probabilities_by_name = combined_attention_vectors


# def find_non_zero_spans(tokens_map, attention_vector_relu):
#   nonzeros = np.argwhere(attention_vector_relu > 0.001)[:, 0]
#   return np.unique([tokens_map.sentence_at_index(i) for i in nonzeros], axis=0)
#
#
# find_sentences_with_attention = find_non_zero_spans


def find_first_gt(indx: int, indices) -> int or None:
  gts = [i for i in indices if i > indx]
  if gts:
    return min(gts)
  return None


def remove_colliding_spans(spans, eps=0):
  if len(spans) == 0:
    return []
  """
  [0..2][2..4][4..5] -> [4..5]
  """
  ret = []
  for i in range(0, len(spans) - 1):
    distance_to_next = abs(spans[i][1] - spans[i + 1][0])
    if distance_to_next > eps:  # sections is not empty
      ret.append(spans[i])
  ret.append(spans[-1])
  return np.array(ret)


# def merge_colliding_spans(spans: Spans, eps=0) -> Spans:
#
#   sorted_spans = sorted(spans, lambda x:x[0])

# def span_intersect

def merge_colliding_spans(spans: Spans, eps=0) -> Spans:
  sorted_spans = sorted(spans, key=lambda x: x[0])
  sorted_spans = [[s[0], s[1]] for s in sorted_spans]
  if len(sorted_spans) == 0:
    return []
  """
  [0..2][2..4][4..5] -> [0..5]
  """
  ret = [sorted_spans[0]]

  for i in range(1, len(sorted_spans)):
    span_a = ret[-1]
    span_b = sorted_spans[i]

    distance_to_next = span_b[0] - span_a[1]
    if distance_to_next > eps:  # sections is not empty
      ret.append(span_b)
    else:
      # merge
      span_a[1] = span_b[1]

  ret = [(s[0], s[1]) for s in ret]
  return np.array(ret)


#
# def per_token_similarity_cosine(text_emb, pattern_emb):
#   a_distances = np.zeros(len(text_emb))
#   for p in pattern_emb:
#     t_distances = 1 - dist_cosine_to_point(text_emb, p)
#     a_distances = sum_probabilities([t_distances, a_distances])
#   return a_distances


def calc_distances_to_pattern(sentences_embeddings_: FixedVectors, pattern_embedding: FixedVector,
                              dist_func=distance.cosine) -> FixedVector:
  # assert len(pattern_embedding.shape) == 1
  # assert len(sentences_embeddings_.shape) == 2
  # assert sentences_embeddings_.shape[1] == pattern_embedding.shape[0]

  _distances = np.ones(len(sentences_embeddings_))
  for word_index in range(0, len(sentences_embeddings_)):
    _distances[word_index] = max(0, min(1, 1.0 - dist_func(sentences_embeddings_[word_index], pattern_embedding)))

  return _distances


import pandas as pd


def calc_distances_per_pattern(sentences_embeddings_: [], patterns_named_embeddings: DataFrame) -> DataFrame:
  # TODO: see https://keras.io/layers/merge/#dot

  distances_per_pattern_dict = pd.DataFrame()
  # print(distances_per_pattern_dict)
  for i, col in patterns_named_embeddings.iteritems():
    # print(col.name, col.values)
    _distances = calc_distances_to_pattern(sentences_embeddings_, col.values)
    distances_per_pattern_dict[col.name] = _distances

  return distances_per_pattern_dict


def calc_distances_per_pattern_dict(sentences_embeddings_: [], patterns_names: [], patterns_embeddings: [[float]]):
  # TODO: see https://keras.io/layers/merge/#dot
  # TODO: use pandas dataframes
  warnings.warn("use calc_distances_per_pattern", DeprecationWarning)
  # assert len(patterns_names) == len(patterns_embeddings)
  distances_per_pattern_dict = {}
  for i in range(len(patterns_names)):
    _distances = calc_distances_to_pattern(sentences_embeddings_, patterns_embeddings[i])

    pattern = patterns_names[i]
    distances_per_pattern_dict[pattern[0]] = _distances

  return distances_per_pattern_dict


def spans_between_non_zero_attention(attention_v: FixedVector):
  # finding non-zero attention points (after ReLu)
  q_sent_indices = np.nonzero(attention_v)[0]

  q_sent_spans = []
  if len(q_sent_indices) > 0:
    q_sent_spans.append(slice(None, q_sent_indices[0]))
    ## re-group indices by spans question slices
    q_sent_spans += [slice(q_sent_indices[i], q_sent_indices[i + 1]) for i in range(len(q_sent_indices) - 1)]
    ## add last segment
    q_sent_spans.append(slice(q_sent_indices[-1], None))

  # print(q_sent_spans)
  return q_sent_spans


def best_above(v: FixedVector, relu_threshold=0.5) -> FixedVector:
  x = relu(v, relu_threshold)
  _nonzeros = np.nonzero(x)

  if len(_nonzeros[0]) > 0:
    _min = np.min(x[_nonzeros])
    _max = max(x)
    _med = (_min + _max) / 2
    return relu(x, _med - 0.0001)
  else:
    return x


def spans_to_attention(spans: [[int]], length: int) -> FixedVector:
  selection = np.zeros(length)
  for span in list(spans):
    selection[span[0]:span[1]] += 1

  return selection


def _find_max_xy_in_matrix(vals):
  max_x = 0
  max_y = 0
  maxval = 0
  for x in range(vals.shape[0]):

    for y in range(vals.shape[1]):
      val = vals[x][y]
      if val > maxval:
        max_x = x
        max_y = y
        maxval = val

  return max_x, max_y, maxval


def attribute_patternmatch_to_index(header_to_pattern_distances_: pd.DataFrame,
                                    threshold=HyperParameters.header_topic_min_confidence):
  vals = header_to_pattern_distances_.values
  # print(header_to_pattern_distances_)
  headers_n = vals.shape[0]
  # print('headers_n', headers_n)

  pairs = []
  for __header_index in range(headers_n):

    header_index, pattern_index, maxval = _find_max_xy_in_matrix(vals)

    pattern_name = header_to_pattern_distances_.columns[pattern_index]
    max_pair = ((pattern_name, header_index), maxval)

    if maxval > threshold:
      pairs.append(max_pair)

    vals[:, pattern_index] = -1
    vals[header_index, :] = -1

  return pairs


def attention_vector(pattern_emb, text_emb: Embeddings) -> FixedVector:
  return np.array([1.0 - distance.cosine(e, pattern_emb) for e in text_emb])


def multi_attention_vector(patterns_emb: Embeddings, text_emb: Embeddings) -> FixedVector:
  vectors: FixedVectors = []
  for pattern_emb in patterns_emb:
    av = attention_vector(pattern_emb, text_emb)
    vectors.append(av)

  return max_exclusive_pattern(vectors)


def get_centroids(embeddings: Embeddings, clustered: pd.DataFrame, labels_column: str) -> Embeddings:
  centroids = []
  for cn in np.unique(clustered[labels_column]):
    items = clustered[clustered[labels_column] == cn]
    m = embeddings[items.index].mean(axis=0)
    centroids.append(m)

  return np.array(centroids)


def softmax_rows(headers_df: DataFrame, columns):
  _x = headers_df[columns]
  _x = scs.softmax(_x, axis=1)
  headers_df[columns] = _x

  return headers_df


def is_in(c: int, span: [int]):
  return c >= span[0] and c < span[1]


def is_span_intersect(span1, span2) -> bool:
  '''
  ......a.....b....
  ....c....d.......
  :param span1:
  :param span2:
  :return:
  '''
  return is_in(span1[0], span2) or is_in(span1[1], span2) \
         or is_in(span2[0], span1) or is_in(span2[1], span1)
