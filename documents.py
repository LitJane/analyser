import sys
import traceback

from text_tools import untokenize, replace_tokens, tokenize_text

TEXT_PADDING_SYMBOL = ' '

import warnings
import os, pickle
from text_tools import Tokens, my_punctuation

import nltk

nltk.download('punkt')


class TextMap:

  def __init__(self, text: str, map=None):
    self._full_text = text
    self._offset_chars = 0
    if map is None:
      self.map = TOKENIZER_DEFAULT.tokens_map(text)
    else:
      # if len(map)<1:
      #   raise RuntimeError('Cannot deal with empty tokenization map')

      self.map = list(map)

    self.untokenize = self.text_range  # alias

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

  def set_token(self, index, new_token):
    assert len(new_token) == self.map[index][1] - self.map[index][0]
    self._full_text = self._full_text[: self.map[index][0]] + new_token + self._full_text[self.map[index][1]:]

  def finditer(self, regexp):
    for m in regexp.finditer(self.text):
      yield self.token_indices_by_char_range_2(m.span(0))

  def token_index_by_char(self, _char_index: int) -> int:
    """
    [span 0] out of span [span 1] [span 2]

    :param char_index:
    :return:
    """

    char_index = _char_index + self._offset_chars
    for span_index in range(len(self.map)):
      span = self.map[span_index]
      if char_index < span[1]:  # span end
        return span_index

    if char_index >= len(self.text):
      return len(self.map)
    return -1

  #     for span_index in range(len(self.map)):
  #       span = self.map[span_index]
  #       if char_index < span[1]:
  #         return span_index
  # TODO: error is here!!!

  def token_indices_by_char_range_2(self, span: [int]) -> (int, int):
    a = self.token_index_by_char(span[0])
    b = self.token_index_by_char(span[1])
    if a == b and b >= 0:
      b = a + 1
    return (a, b)

  def token_indices_by_char_range(self, span: [int]) -> slice:
    warnings.warn("use token_indices_by_char_range_2", DeprecationWarning)
    a = self.token_index_by_char(span[0])
    b = self.token_index_by_char(span[1])
    if a == b and b >= 0:
      b = a + 1
    return slice(a, b)

  def slice(self, span: slice) -> 'TextMap':
    sliced = TextMap(self._full_text, self.map[span])
    if sliced.map:
      sliced._offset_chars = sliced.map[0][0]
      sliced._full_text = sliced._full_text[0:  sliced.map[-1][-1]]
    else:
      sliced._offset_chars = 0
    # first_char_index = sliced.map[0][0]
    # for _s in sliced.map:
    #   _s[0] -= first_char_index
    #   _s[1] -= first_char_index
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
        yield [last, i + addon]
        last = i + 1
    yield [last, self.get_len()]

  def sentence_at_index(self, i: int) -> (int, int):
    sent_spans = self.split_spans('\n', add_delimiter=True)
    for s in sent_spans:
      if i >= s[0] and i < s[1]:
        return s
    return [0, self.get_len()]

  def text_range(self, span) -> str:
    if span[0] >= len(self.map):
      return ''

    try:
      start = self.map[span[0]][0]
      _last = min(len(self.map), span[1])
      stop = self.map[_last - 1][1]

      # assume map is ordered
      return self._full_text[start: stop]
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

  def __getitem__(self, key):
    if isinstance(key, slice):
      # Get the start, stop, and step from the slice
      return [self[ii] for ii in range(*key.indices(len(self)))]
    elif isinstance(key, int):

      r = self.map[key]
      # print('__getitem__', key)
      return self._full_text[r[0]:r[1]]
    else:
      raise TypeError("Invalid argument type.")

  def get_tokens(self):
    return [
      self._full_text[tr[0]:tr[1]] for tr in self.map
    ]

  tokens = property(get_tokens)
  text = property(get_text)


class CaseNormalizer:
  __shared_state = {}  ## see http://code.activestate.com/recipes/66531/

  def __init__(self):
    self.__dict__ = self.__shared_state
    if 'replacements_map' not in self.__dict__:
      __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
      p = os.path.join(__location__, 'vocab', 'word_cases_stats.pickle')
      print('loading word cases stats model', p)

      with open(p, 'rb') as handle:
        self.replacements_map = pickle.load(handle)

  def normalize_tokens_map_case(self, map: TextMap) -> TextMap:
    norm_tokens = replace_tokens(map.tokens, self.replacements_map)
    norm_map = TextMap(map._full_text, map.map)
    for k in range(len(map)):
      norm_map.set_token(k, norm_tokens[k])
    # chars = list(map.text)
    # for i in range(0, len(map)):
    #   r = map.map[i]
    #   chars[r[0]:r[1]] = norm_tokens[i]
    # norm_map = TextMap(''.join(chars), list(map.map))
    # # XXXX
    # dfdfdfdf
    return norm_map

  def normalize_tokens(self, tokens: Tokens) -> Tokens:
    return replace_tokens(tokens, self.replacements_map)

  def normalize_text(self, text: str) -> str:
    warnings.warn(
      "Deprecated, because this class must not perform tokenization. Use normalize_tokens or  normalize_tokens_map_case",
      DeprecationWarning)
    tokens = tokenize_text(text)
    tokens = self.normalize_tokens(tokens)
    return untokenize(tokens)

  def normalize_word(self, token: str) -> str:
    if token.lower() in self.replacements_map:
      return self.replacements_map[token.lower()]
    else:
      return token


class EmbeddableText:
  warnings.warn("deprecated", DeprecationWarning)

  def __init__(self):
    warnings.warn("deprecated", DeprecationWarning)

    self.embeddings = None


# ---------------------------------------------------


class GTokenizer:
  def tokenize(self, s) -> Tokens:
    raise NotImplementedError()

  def untokenize(self, t: Tokens) -> str:
    raise NotImplementedError()


def span_tokenize(text):
  start_from = 0
  for token in nltk.word_tokenize(text):
    if token == "''":
      token = '"'

    if token == "``":
      token = '"'

    ix_new = text.find(token, start_from)
    if ix_new < 0:
      print(f'ACHTUNG! [{token}] not found with text.find, next text is: {text[start_from:start_from + 30]}')
    else:
      start_from = ix_new
      end = start_from + len(token)
      yield [start_from, end]
      start_from = end


class DefaultGTokenizer(GTokenizer):

  def __init__(self):

    # pth = os.path.join(os.path.dirname(__file__), 'nltk_data_download')
    # nltk.download('punkt', download_dir=pth)
    pass

  def tokenize_line(self, line):
    return [line[t[0]:t[1]] for t in span_tokenize(line)]

  def tokenize(self, text) -> Tokens:
    return [text[t[0]:t[1]] for t in self.tokens_map(text)]

  def untokenize(self, tokens: Tokens) -> str:
    warnings.warn("deprecated", DeprecationWarning)
    # TODO: remove it!!
    return "".join([" " + i if not i.startswith("'") and i not in my_punctuation else i for i in tokens]).strip()

  # build tokens map to char pos
  def tokens_map(self, text):

    result = []
    for i in range(len(text)):
      if text[i] == '\n':
        result.append([i, i + 1])

    result += [s for s in span_tokenize(text)]

    result.sort(key=lambda x: x[0])
    return result


# TODO: use it!
TOKENIZER_DEFAULT = DefaultGTokenizer()
