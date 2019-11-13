import math
import warnings
from typing import List, TypeVar, Iterable, Generic

import numpy as np
import scipy.spatial.distance as distance

from documents import TextMap
from text_tools import Tokens

FixedVector = TypeVar('FixedVector', List[float], np.ndarray)
Vector = TypeVar('Vector', FixedVector, Iterable[float])
Vectors = TypeVar('Vectors', List[Vector], Iterable[Vector])
FixedVectors = TypeVar('FixedVectors', List[FixedVector], Iterable[FixedVector])

T = TypeVar('T')  # just type variable


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

  if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

  s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
  # print(len(s))
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
  assert type(x) is np.ndarray

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
  for i in range(len(x)):
    m += x[i]
    innertia[i] = m
    m *= decay

  return innertia


def momentum(x: FixedVector, decay=0.999) -> np.ndarray:
  innertia = np.zeros(len(x))
  m = 0
  for i in range(len(x)):
    m = max(m, x[i])
    innertia[i] = m
    m *= decay

  return innertia


def momentum_t(x: FixedVector, half_decay: int = 10, left=False) -> np.ndarray:
  assert half_decay > 0

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
  assert half_decay > 0

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

  assert _sum is not None

  return _sum


def filter_values_by_key_prefix(dictionary: dict, prefix: str) -> Vectors:
  for p in dictionary:
    if str(p).startswith(prefix):
      yield dictionary[p]


def max_exclusive_pattern_by_prefix(distances_per_pattern_dict, prefix):
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
    assert len(tokens) == len(attention)
    self.tokens = tokens
    self.attention = attention


class SemanticTag:
  def __init__(self, kind, value, span: (int, int), span_map='words'):
    self.kind = kind
    self.value = value
    '''name of the parent (or group) tag '''
    self.parent: str or None = None
    if span:
      self.span = (int(span[0]), int(span[1]))  # kind of type checking
    else:
      self.span = (0, 0)  # TODO: might be keep None?
    self.span_map = span_map
    self.confidence = 1.0
    self.display_value = value

  def as_slice(self):
    return slice(self.span[0], self.span[1])

  def isNotEmpty(self) -> bool:
    return self.span is not None and self.span[0] != self.span[1]

  @staticmethod
  def find_by_kind(list: List['SemanticTag'], kind: str) -> 'SemanticTag':
    for s in list:
      if s.kind == kind:
        return s

  def offset(self, span_add: int):
    self.span = self.span[0] + span_add, self.span[1] + span_add

  def is_nested(self, other: [int]) -> bool:
    return self.span[0] <= other[0] and self.span[1] >= other[1]

  def __str__(self):
    return f'SemanticTag: {self.kind} {self.span} {self.value} {self.display_value}  {self.confidence}'

  def quote(self, tm: TextMap):
    return tm.text_range(self.span)

  slice = property(as_slice)


def estimate_confidence(vector: FixedVector) -> (float, float, int, float):
  assert vector is not None
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


def select_most_confident_if_almost_equal(a: ProbableValue, alternative: ProbableValue,
                                          equality_range=0.0) -> ProbableValue:
  try:
    if abs(a.value.value - alternative.value.value) < equality_range:
      if a.confidence > alternative.confidence:
        return a
      else:
        return alternative
  except:
    # TODO: why dan hell we should have an exception here??
    return a

  return a


def combined_attention_vectors(vectors_dict, vector_names):
  vectors = [vectors_dict[v] for v in vector_names]
  return sum_probabilities(vectors)


sum_probabilities_by_name = combined_attention_vectors


def find_non_zero_spans(tokens_map, attention_vector_relu):
  nonzeros = np.argwhere(attention_vector_relu > 0.001)[:, 0]
  return np.unique([tokens_map.sentence_at_index(i) for i in nonzeros], axis=0)


find_sentences_with_attention = find_non_zero_spans


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


def merge_colliding_spans(spans, eps=0):
  if len(spans) == 0:
    return []
  """
  [0..2][2..4][4..5] -> [0..5]
  """
  ret = [spans[0]]

  for i in range(1, len(spans)):
    distance_to_next = abs(ret[-1][1] - spans[i][0])
    if distance_to_next > eps:  # sections is not empty
      ret.append(spans[i])
    else:
      ret[-1][1] = spans[i][1]

  return np.array(ret)


#
# def per_token_similarity_cosine(text_emb, pattern_emb):
#   a_distances = np.zeros(len(text_emb))
#   for p in pattern_emb:
#     t_distances = 1 - dist_cosine_to_point(text_emb, p)
#     a_distances = sum_probabilities([t_distances, a_distances])
#   return a_distances


def calc_distances_to_pattern(sentences_embeddings_, pattern_embedding, dist_func=distance.cosine):
  assert len(pattern_embedding.shape) == 1
  assert len(sentences_embeddings_.shape) == 2
  assert sentences_embeddings_.shape[1] == pattern_embedding.shape[0]

  _distances = np.ones(len(sentences_embeddings_))
  for word_index in range(0, len(sentences_embeddings_)):
    _distances[word_index] = max(0, min(1, 1.0 - dist_func(sentences_embeddings_[word_index], pattern_embedding)))

  return _distances


def calc_distances_per_pattern_dict(sentences_embeddings_, patterns_dict, patterns_embeddings):
  # TODO: see https://keras.io/layers/merge/#dot

  distances_per_pattern_dict = {}
  for i in range(len(patterns_dict)):
    pattern = patterns_dict[i]
    _distances = calc_distances_to_pattern(sentences_embeddings_, patterns_embeddings[i])

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


def spans_to_attention(spans: [[int]], length: int) -> FixedVector:
  selection = np.zeros(length)
  for span in list(spans):
    selection[span[0]:span[1]] += 1

  return selection


def sentences_attention_to_words(attention_v, sentence_map: TextMap, words_map: TextMap):
  q_sent_indices = np.nonzero(attention_v)[0]
  w_spans_attention = np.zeros(len(words_map))
  char_ranges = [(sentence_map.map[i], attention_v[i]) for i in q_sent_indices]

  w_spans = []
  for char_range, a in char_ranges:
    words_range = words_map.token_indices_by_char_range(char_range)
    w_spans.append(words_range)
    w_spans_attention[words_range[0]:words_range[1]] += a

  return w_spans, w_spans_attention
