import hashlib
import os
import pickle
import sys
import traceback
import warnings

import nltk

from analyser.hyperparams import models_path
from analyser.log import logger

nltk.data.path.append(os.path.join(models_path, 'nltk'))
import numpy as np

from analyser.ml_tools import spans_to_attention, FixedVector
from analyser.text_tools import Tokens, untokenize, replace_tokens, split_into_sentences

TEXT_PADDING_SYMBOL = ' '


# nltk.download('punkt')


class TextMap:

  def __init__(self, text: str, map=None):
    self._full_text = str(text)
    self._offset_chars = 0
    if map is None:
      self.map = TOKENIZER_DEFAULT.tokens_map(text)
      self.cleanup()
    else:
      self.map = list(map)

    self.untokenize = self.text_range  # alias

  def getcopy(self):
    tm = TextMap(self._full_text, self.map)
    tm._offset_chars = self._offset_chars
    return tm

  def get_full_text(self):
    return str(self._full_text)

  def cleanup(self):
    warnings.warn("fix tokenization instead of using it, also, it breaks attributes mapping", DeprecationWarning)
    '''
    removes empty span (for obsolete docs
    :return:
    '''

    self.map = [x for x in self.map if x[0] != x[1]]

  def __add__(self, other):
    off = len(self._full_text)
    self._full_text += other._full_text
    # if len(self.map)>0:
    #   off = self.map[-1][-1]
    # else:
    #   off=len(self._full_text)
    for span in other.map:
      self.map.append((span[0] + off, span[1] + off))

    return self

  def regex_attention(self, regex)-> FixedVector:
    matches = list(self.finditer(regex))
    return spans_to_attention(matches, len(self))

  def set_token(self, index, new_token):
    span = self.map[index]
    tlen = span[1] - span[0]
    if len(new_token) == tlen:
      self._full_text = self._full_text[: span[0]] + new_token + self._full_text[span[1]:]
    else:
      et = self.text_range(span)
      q = self.text_range((span[0] - 10, span[1] + 10))
      msg = f'index:{index}; len of replacement [{new_token}] {len(new_token)} is not equal to len of existing token [{et}] {tlen}, quote="{q}"'
      raise AssertionError(msg)

  def finditer(self, regexp):
    for m in regexp.finditer(self.text):
      yield self.token_indices_by_char_range(m.span(0))

  def token_index_by_char(self, _char_index: int) -> int:
    if not self.map: return -1

    local_off = self.map[0][0] - self._offset_chars
    """
    [span 0] out of span [span 1] [span 2]

    :param _char_index:
    :return:
    """

    char_index = local_off + _char_index + self._offset_chars
    for span_index in range(len(self.map)):
      span = self.map[span_index]
      if char_index < span[1]:  # span end
        return span_index

    if char_index >= len(self.text):
      return len(self.map)
    return -1

  def token_indices_by_char_range(self, char_span: [int]) -> (int, int):
    a = self.token_index_by_char(char_span[0])
    b = self.token_index_by_char(char_span[1])
    if a == b and b >= 0:
      b = a + 1
    return a, b

  def slice(self, span: slice) -> 'TextMap':
    sliced = TextMap(self._full_text, self.map[span])
    if sliced.map:
      sliced._offset_chars = sliced.map[0][0]
      sliced._full_text = sliced._full_text[0:  sliced.map[-1][-1]]
    else:
      sliced._offset_chars = 0

    return sliced

  def split(self, delimiter: str) -> [Tokens]:
    last = 0

    for i in range(self.get_len()):
      if self[i] == delimiter:
        yield self[last: i]
        last = i + 1
    yield self[last: self.get_len()]

  def split_spans(self, delimiter: str, add_delimiter=False):
    addon = 0
    if add_delimiter:
      addon = 1

    last = 0
    for i in range(self.get_len()):
      if self[i] == delimiter:
        yield (last, i + addon)
        last = i + 1
    yield (last, self.get_len())

  def sentence_at_index(self, i: int, return_delimiters=True) -> (int, int):

    warnings.warn("use LegalDocument.sentence_at_index and sentence map", DeprecationWarning)

    sent_spans = self.split_spans('\n', add_delimiter=return_delimiters)
    d_add = 1
    if return_delimiters:
      d_add = 0

    for s in sent_spans:
      if i >= s[0] and i < s[1] + d_add:
        return s
    return (0, self.get_len())

  def char_range(self, span: [int]) -> (int, int):
    a = span[0]
    b = span[1]

    if a is None:
      a = 0

    if b is None:
      b = len(self.map)

    if a >= len(self.map):
      return 0, 0

    start = self.map[a][0]

    _last = min(len(self.map), b)
    stop = self.map[_last - 1][1]

    return start - self._offset_chars, stop - self._offset_chars

  def remap_spans(self, spans, target_map: 'TextMap'):
    if self._full_text != target_map._full_text:
      raise ValueError("_full_text != target_map._full_text")
    ret = []
    for span in spans:
      char_range = self.char_range(span)
      words_range = target_map.token_indices_by_char_range(char_range)
      ret.append(words_range)
    return ret

  def remap_span(self, span, target_map: 'TextMap'):
    if self._full_text != target_map._full_text:
      msg = f'remap: texts differ {len(self._full_text)}!={len(target_map._full_text)}'
      raise ValueError(msg)

    char_range = self.char_range(span)
    target_range = target_map.token_indices_by_char_range(char_range)
    return target_range

  def remap_slices(self, spans, target_map: 'TextMap'):
    assert self._full_text == target_map._full_text
    ret = []
    for span in spans:
      char_range = self.char_range([span.start, span.stop])
      words_range = target_map.token_indices_by_char_range(char_range)
      ret.append(words_range)
    return ret

  def text_range(self, span) -> str:
    try:
      start, stop = self.char_range(span)
      return self._full_text[start + self._offset_chars: stop + self._offset_chars]
    except:
      err = f'cannot deal with {span}'
      traceback.print_exc(file=sys.stdout)
      raise RuntimeError(err)

  def get_text(self):
    if len(self.map) == 0:
      return ''
    return self.text_range([0, len(self.map)])

  def tokens_by_range(self, span) -> Tokens:
    tokens_i = self.map[span[0]:span[1]]
    return [
      self._full_text[tr[0]:tr[1]] for tr in tokens_i
    ]

  def get_len(self):
    return len(self.map)

  def __len__(self):
    return self.get_len()

  def __getitem__(self, key)->str:
    if isinstance(key, slice):
      # Get the start, stop, and step from the slice
      return [self[ii] for ii in range(*key.indices(len(self)))]
    elif isinstance(key, int):

      r = self.map[key]
      return self._full_text[r[0]:r[1]]
    else:
      raise TypeError("Invalid argument type.")

  def get_tokens(self) -> Tokens:
    return [
      self._full_text[tr[0]:tr[1]] for tr in self.map
    ]

  tokens: Tokens = property(get_tokens, None)
  text: str = property(get_text, None)

  def get_checksum(self):
    _reconstructed = '|'.join(self.tokens).encode('utf-8')
    return hashlib.md5(_reconstructed).hexdigest()


def split_sentences_into_map(substr, max_len_chars=150) -> TextMap:
  spans1 = split_into_sentences(substr, max_len_chars)
  tm = TextMap(substr, spans1)
  return tm


class CaseNormalizer:
  __shared_state = {}  ## see http://code.activestate.com/recipes/66531/

  def __init__(self):
    self.__dict__ = self.__shared_state
    if 'replacements_map' not in self.__dict__:
      p = os.path.join(models_path, 'word_cases_stats.pickle')
      logger.info(f'loading word cases stats model from: {p}')

      with open(p, 'rb') as handle:
        self.replacements_map: dict = pickle.load(handle)

      # selfcheck
      mn = {}
      for k, v in self.replacements_map.items():
        if len(k) != len(v):
          raise ValueError('cannot replace token with one of a diff. length')

        if len(k) > 1:
          mn[k] = v
      self.replacements_map = mn

  def normalize_tokens_map_case(self, tmap: TextMap) -> TextMap:
    norm_tokens = replace_tokens(tmap.tokens, self.replacements_map)
    norm_map: TextMap = tmap.getcopy()

    if len(norm_tokens) != len(tmap.tokens):
      raise ValueError('len(norm_tokens) != len(tmap.tokens)')

    for k in range(len(tmap)):
      span = tmap.map[k]
      if span[0] != span[1] > 0:
        norm_map.set_token(k, norm_tokens[k])
      else:
        msg = f'empty span at index {k}'
        warnings.warn(msg)

    return norm_map

  def normalize_tokens(self, tokens: Tokens) -> Tokens:
    return replace_tokens(tokens, self.replacements_map)

  def normalize_text(self, text: str) -> str:
    warnings.warn(
      "Deprecated, because this class must not perform tokenization. Use normalize_tokens or  normalize_tokens_map_case",
      DeprecationWarning)
    tm = TextMap(text)
    tokens = tm.tokens
    tokens = self.normalize_tokens(tokens)
    return untokenize(tokens)

  def normalize_word(self, token: str) -> str:
    if token.lower() in self.replacements_map:
      return self.replacements_map[token.lower()]
    else:
      return token


class GTokenizer:

  def tokenize(self, s) -> Tokens:
    raise NotImplementedError()


class DefaultGTokenizer(GTokenizer):

  def __init__(self):

    # pth = os.path.join(os.path.dirname(__file__), 'nltk_data_download')
    # nltk.download('punkt', download_dir=pth)
    pass

  def span_tokenize(self, text):
    start_from = 0
    text = text.replace('`', '!')
    text = text.replace('"', '!')
    tokens = list(nltk.word_tokenize(text, language='russian'))
    __debug = []

    for search_token in tokens:

      ix_new = text.find(search_token, start_from)
      if ix_new < 0:
        msg = f'ACHTUNG! [{search_token}] not found with text.find, next text is: {text[start_from:start_from + 30]}'
        warnings.warn(msg)
      else:
        start_from = ix_new
        end = start_from + len(search_token)
        __debug.append((search_token, start_from, end))
        yield [start_from, end]
        start_from = end

  def tokenize_line(self, line):
    return [line[t[0]:t[1]] for t in self.span_tokenize(line)]

  def tokenize(self, text) -> Tokens:
    return [text[t[0]:t[1]] for t in self.tokens_map(text)]

  # build tokens map to char pos
  def tokens_map(self, text):

    result = []
    for i in range(len(text)):
      if text[i] == '\n':
        result.append([i, i + 1])

    result += [s for s in self.span_tokenize(text)]

    result.sort(key=lambda x: x[0])
    return result


TOKENIZER_DEFAULT = DefaultGTokenizer()


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
