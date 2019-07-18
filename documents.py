from text_tools import untokenize, replace_tokens, tokenize_text, Tokens

TEXT_PADDING_SYMBOL = ' '

import warnings
import os, pickle


class TextMap:

  def __init__(self, text: str, map=None):
    self._full_text = text
    if map is None:
      self.map = TOKENIZER_DEFAULT.tokens_map(text)
    else:
      self.map = map

    self.untokenize = self.text_range  # alias

  def token_index_by_char(self, index: int) -> int:
    for i in range(len(self.map)):
      span = self.map[i]
      if index < span[1]:
        return i

    return -1

  def tokens_in_range(self, span: [int]) -> slice:

    a = self.token_index_by_char(span[0])
    b = self.token_index_by_char(span[1])
    return slice(a, b)

  def slice(self, span: slice):
    return TextMap(self._full_text, self.map[span])

  def split(self, delimiter: str) -> [Tokens]:
    last = 0
    for i in range(self.get_len()):
      if self[i] == delimiter:
        yield self[last: i]
        last = i + 1
    yield self[last: self.get_len()]

  def split_spans(self, delimiter: str) -> [Tokens]:
    last = 0
    for i in range(self.get_len()):
      if self[i] == delimiter:
        yield [last, i]
        last = i + 1
    yield [last, self.get_len()]

  def text_range(self, span) -> str:
    start = self.map[span[0]][0]
    _last = min(len(self.map), span[1])
    stop = self.map[_last - 1][1]

    # assume map is ordered
    return self._full_text[start: stop]

  def get_text(self):
    return self.text_range([0, len(self.map)])

  def tokens_in_range(self, span) -> Tokens:
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
    chars = list(map.text)
    for i in range(0, len(map)):
      r = map.map[i]
      chars[r[0]:r[1]] = norm_tokens[i]
    norm_map = TextMap(''.join(chars), map.map)
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


class TokenizedText:

  def __init__(self):
    warnings.warn("deprecated", DeprecationWarning)
    super().__init__()

    self.tokens_cc = None
    self.tokens: Tokens = None

  def get_len(self):
    return len(self.tokens)

  def untokenize(self):
    warnings.warn("deprecated", DeprecationWarning)
    return untokenize(self.tokens)

  def untokenize_cc(self):
    warnings.warn("deprecated", DeprecationWarning)
    return untokenize(self.tokens_cc)

  def concat(self, doc: "TokenizedText"):
    warnings.warn("deprecated", DeprecationWarning)
    self.tokens += doc.tokens
    self.categories_vector += doc.categories_vector
    if self.tokens_cc:
      self.tokens_cc += doc.tokens_cc

  def trim(self, sl: slice):
    warnings.warn("deprecated", DeprecationWarning)
    self.tokens = self.tokens[sl]
    if self.tokens_cc:
      self.tokens_cc = self.tokens_cc[sl]
    self.categories_vector = self.categories_vector[sl]


class EmbeddableText:
  warnings.warn("deprecated", DeprecationWarning)

  def __init__(self):
    warnings.warn("deprecated", DeprecationWarning)

    self.embeddings = None


class MarkedDoc(TokenizedText):

  def __init__(self, tokens, categories_vector):
    super().__init__()

    self.tokens = tokens
    self.categories_vector = categories_vector

  def filter(self, filter_op):
    new_tokens = []
    new_categories_vector = []

    for i in range(self.get_len()):
      _tuple = filter_op(self.tokens[i], self.categories_vector[i])

      if _tuple is not None:
        new_tokens.append(_tuple[0])
        new_categories_vector.append(_tuple[1])

    self.tokens = new_tokens
    self.categories_vector = new_categories_vector


# ---------------------------------------------------

from text_tools import Tokens, my_punctuation


class GTokenizer:
  def tokenize(self, s) -> Tokens:
    raise NotImplementedError()

  def untokenize(self, t: Tokens) -> str:
    raise NotImplementedError()


import nltk


def span_tokenize(text):
  ix = 0
  for word_token in nltk.word_tokenize(text):
    ix = text.find(word_token, ix)
    end = ix + len(word_token)
    yield (ix, end)
    ix = end


class DefaultGTokenizer(GTokenizer):

  def __init__(self):
    nltk.download('punkt')

  def tokenize_line(self, line):
    return [line[t[0]:t[1]] for t in span_tokenize(line)]

  def tokenize(self, text) -> Tokens:
    return [text[t[0]:t[1]] for t in self.tokens_map(text)]

  def untokenize(self, tokens: Tokens) -> str:
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
