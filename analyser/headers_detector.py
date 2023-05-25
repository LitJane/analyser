import numpy as np
import pandas as pd

from analyser.doc_structure import get_tokenized_line_number
from analyser.documents import TextMap
from analyser.legal_docs import PARAGRAPH_DELIMITER
from analyser.text_tools import Tokens, _count_capitals, _count_digits


def doc_features(tokens_map: TextMap):
  body_lines_ranges = tokens_map.split_spans(PARAGRAPH_DELIMITER, add_delimiter=True)

  _doc_features = []
  _line_spans = []
  ln = 0
  _prev_features = None
  for line_span in body_lines_ranges:
    _line_spans.append(line_span)

    _features = line_features(tokens_map, line_span, ln, _prev_features)
    _doc_features.append(_features)
    _prev_features = _features
    ln += 1
  doc_featuresX_data = pd.DataFrame.from_records(_doc_features)
  doc_features_data = np.array(doc_featuresX_data)

  return doc_features_data, _line_spans


def _onehot(x: bool or int) -> float:
  if x:
    return 1.0
  else:
    return 0.0


def _has_symbols(txt: str, strange_symbols) -> float:
  for c in strange_symbols:
    if txt.count(c) > 0:
      return 1.0
  return 0.0


def get_token_features(token: str):
  features = {
    'isdigit': _onehot(token.isdigit()),
    'istitle': _onehot(token.istitle()),
    'nl': _onehot(token in ('\n', '\r')),

    'has_underscore': _has_symbols(token, '_'),

    'isupper': _onehot(token.isupper()),
    'has_quotes': _has_symbols(token, '«»"\'"'),
    'has_trash': _has_symbols(token, '[]$@+^&(){}<>'),
    'has_syntax': _has_symbols(token, '!.:;,'),

    'isalpha': _onehot(token.isalpha()),
    'len_lt3': _onehot(len(token) < 3),
    'len_lt2': _onehot(len(token) < 2),
    'islower': _onehot(token.islower()),
    'isprintable': _onehot(token.isprintable()),

  }
  return features


def get_tokens_features(tokens):
  _doc_features = []

  for t in tokens:
    _features = get_token_features(t)
    _doc_features.append(_features)

  doc_featuresX_data = pd.DataFrame.from_records(_doc_features)
  doc_featuresX_data['_reserved'] = 0.0
  doc_featuresX_data['h'] = 0.0

  position_enc = np.arange(1, 0, -0.0005)
  max_pos = min(len(position_enc), len(doc_featuresX_data['h']))
  doc_featuresX_data['h'][0:max_pos] = position_enc[0:max_pos]

  return doc_featuresX_data


def line_features(tokens_map: TextMap, line_span: (int, int), line_number: int, prev_features):
  tokens: Tokens = tokens_map.tokens_by_range(line_span)
  # TODO: add previous and next lines features
  txt: str = tokens_map.text_range(line_span)

  numbers, span, k, s = get_tokenized_line_number(tokens, 0)
  if not numbers:
    numbers = []
    number_minor = -2
    number_major = -2
  else:
    number_minor = numbers[-1]
    number_major = numbers[0]

  header_id = ' '.join(tokens[span[1]:])
  header_id = header_id.lower()

  all_upper = header_id.upper() == header_id

  features = {
    'line_number': line_number,
    # 'popular': _onehot(header_id in popular_headers),
    # 'cr_count': txt.count('\r'),

    'has_contract': _onehot(txt.lower().find('договор')),
    'has_article': _onehot(txt.lower().find('статья')),
    'all_uppercase': _onehot(all_upper),
    'len_tokens': len(tokens),
    'len_chars': len(txt),
    'number_level': len(numbers),
    'number_minor': number_minor,
    'number_major': number_major,
    'number_roman': _onehot(s),
    'endswith_dot': _onehot(txt.rstrip().endswith('.')),
    'endswith_comma': _onehot(txt.rstrip().endswith(',')),
    'endswith_underscore': _onehot(txt.rstrip().endswith('_')),

    # counts
    'dots': header_id.count('.'),
    'tabs': txt.count('\t'),
    'spaces_inside': txt.strip().count(' '),
    'spaces_all': txt.count(' '),
    'commas': header_id.count(','),
    'brackets': _count_strange_symbols(txt, '(){}[]'),
    'dashes': header_id.count('-'),
    'colons': header_id.count(':'),
    'semicolons': header_id.count(';'),
    'strange_symbols': _count_strange_symbols(header_id, '[$@+]?^&'),
    'capitals': _count_capitals(txt),
    'digits': _count_digits(header_id),
    'quotes': _count_strange_symbols(txt, '«»"\'"'),
    'underscores': _count_strange_symbols(txt, '_')
  }

  return features


def _count_strange_symbols(txt, strange_symbols) -> int:
  res = 0
  for c in strange_symbols:
    res += txt.count(c)
  return res


TOKEN_FEATURES = 2 + len(get_token_features("a"))
