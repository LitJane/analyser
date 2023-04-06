#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import warnings
from typing import List

import numpy as np
import scipy.spatial.distance as distance
from numpy.core.multiarray import ndarray
from pyjarowinkler import distance as jaro

Tokens = List[str] or ndarray


def to_float(string) -> float:
  try:
    return float(str(string).replace(" ", "").replace(",", ".").replace("=", "."))
  except:
    return np.nan


def replace_with_map(txt, replacements):
  a = txt
  for (src, target) in replacements:
    a = a.replace(src, target)

  return a


def remove_empty_lines(original_text):
  a = "\n".join([ll.strip() for ll in original_text.splitlines() if ll.strip()])
  return a.replace('\t', ' ')


# ----------------------------------------------------------------
# DISTANCES
# ----------------------------------------------------------------


def dist_cosine_to_point(text_emb, pt):
  t_distances = np.zeros(len(text_emb))
  for i in range(len(text_emb)):

    d = distance.cosine(text_emb[i], pt)
    if d > 1:
      d = 1
    t_distances[i] = d

  return t_distances


def dist_hausdorff(u, v):
  return max(distance.directed_hausdorff(v, u, 4)[0], distance.directed_hausdorff(u, v, 4)[0]) / 40


def dist_mean_cosine(u, v):
  return distance.cosine(u.mean(0), v.mean(0))


def dist_mean_eucl(u, v):
  return distance.euclidean(u.mean(0), v.mean(0))


def dist_sum_cosine(u, v):
  return distance.cosine(u.sum(0), v.sum(0))


# not in use
def dist_correlation_min_mean(u, v):
  if u.shape[0] > v.shape[0]:
    return distance.cdist(u, v, 'correlation').min(0).mean()
  else:
    return distance.cdist(v, u, 'correlation').min(0).mean()


def dist_cosine_min_mean(u, v):
  if u.shape[0] > v.shape[0]:
    return distance.cdist(u, v, 'cosine').min(0).mean()
  else:
    return distance.cdist(v, u, 'cosine').min(0).mean()


# not in use
def dist_euclidean_min_mean(u, v):
  if u.shape[0] > v.shape[0]:
    return distance.cdist(u, v, 'euclidean').min(0).mean()
  else:
    return distance.cdist(v, u, 'euclidean').min(0).mean()


# ----------------------------------------------------------------
# MISC
# ----------------------------------------------------------------
def split_version(v: str) -> [int]:
  arr = (v + ".0.0.0").replace('v', '').replace('_', '.').split('.')
  return [int(a) for a in arr][:3]


def norm_matrix(mtx):
  mtx = mtx - mtx.min()
  mtx = mtx / np.abs(mtx).max()
  return mtx


def min_index(sums):
  min_i = 0
  min_d = float('Infinity')

  for d in range(0, len(sums)):
    if sums[d] < min_d:
      min_d = sums[d]
      min_i = d

  return min_i


# ----------------------------------------------------------------
# TOKENS
# ----------------------------------------------------------------

def sentence_similarity_matrix(emb, distance_function):
  mtx = np.zeros(shape=(emb.shape[0], emb.shape[0]))

  # TODO: write it Pythonish!!
  for u in range(emb.shape[0]):
    for v in range(emb.shape[0]):
      mtx[u, v] = distance_function(emb[u], emb[v])

  # TODO: no norm here
  return mtx  # norm_matrix(mtx)


my_punctuation = r"""!"#$%&'*+,-./:;<=>?@[\]^_`{|}~"""


def untokenize(tokens: Tokens) -> str:
  warnings.warn("deprecated", DeprecationWarning)
  return "".join([" " + i if not i.startswith("'") and i not in my_punctuation else i for i in tokens]).strip()


def find_token_before_index(tokens: Tokens, index, token, default_ret=-1):
  warnings.warn("deprecated: method must be moved to TextMap class", DeprecationWarning)
  for i in reversed(range(min(index, len(tokens)))):
    if tokens[i] == token:
      return i
  return default_ret


def find_token_after_index(tokens: Tokens, index, token, default_ret=-1):
  warnings.warn("deprecated: method must be moved to TextMap class", DeprecationWarning)
  for i in (range(min(index, len(tokens)))):
    if tokens[i] == token:
      return i
  return default_ret


def hot_quotes(tokens: Tokens) -> (np.ndarray, np.ndarray):
  q_re_open = '\'\"«<{['
  q_re_close = '\'\"»>]'
  _quotes_open = np.zeros(len(tokens))
  _quotes_closing = np.zeros(len(tokens))

  quotes_attention = 1
  for i in range(len(tokens)):
    if tokens[i][0] in q_re_open:
      _quotes_open[i] = quotes_attention
    if tokens[i][0] in q_re_close:
      _quotes_closing[i] = quotes_attention

  return _quotes_open, _quotes_closing


def hot_punkt(tokens: Tokens) -> np.ndarray:
  _hot_punkt = np.zeros(len(tokens))

  for i in range(len(tokens)):
    if tokens[i][0] in my_punctuation + '–«»()[] ':
      _hot_punkt[i] = 1

  return _hot_punkt


def acronym(n):
  return ''.join([x[0] for x in n.split(' ') if len(x) > 1]).upper()


def replace_tokens(tokens: Tokens, replacements_map: dict):
  return [replacements_map.get(t.lower(), t) for t in tokens]


def roman_to_arabic(n) -> int or None:
  roman = n.upper().lstrip()
  if not check_valid_roman(roman):
    return None

  keys = ['IV', 'IX', 'XL', 'XC', 'CD', 'CM', 'I', 'V', 'X', 'L', 'C', 'D', 'M']
  to_arabic = {'IV': '4', 'IX': '9', 'XL': '40', 'XC': '90', 'CD': '400', 'CM': '900',
               'I': '1', 'V': '5', 'X': '10', 'L': '50', 'C': '100', 'D': '500', 'M': '1000'}
  for key in keys:
    if key in roman:
      roman = roman.replace(key, ' {}'.format(to_arabic.get(key)))

  return sum(int(num) for num in roman.split())


def check_valid_roman(roman) -> bool:
  if len(roman.strip()) == 0:
    return False
  invalid = ['IIII', 'VV', 'XXXX', 'LL', 'CCCC', 'DD', 'MMMM', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  if any(sub in roman for sub in invalid):
    return False
  return True


def roman_might_be(wrd) -> int or None:
  try:
    return roman_to_arabic(wrd)
  except:
    return None


def string_to_ip(txt) -> list or None:
  ret = []
  n = txt.split('.')
  for c in n:
    try:
      ret.append(int(c))
    except:
      pass
  return ret


def unquote(s):
  if not s:
    return s
  if len(s) < 2:
    return s

  if s[0] == '«' and s[-1] == '»':
    return s[1:-1]

  if s[0] == '"' and s[-1] == '"':
    return s[1:-1]

  if s[0] == '\'' and s[-1] == '\'':
    return s[1:-1]

  return s


def find_best_sentence_end(txt) -> (int, str):
  delimiters_prio = ['\n', '•', '.!?', ';', ',', "-—", ')', ':', ' ']

  for delimiters in delimiters_prio:

    for i in reversed(range(len(txt))):
      c = txt[i:i + 1]
      if delimiters.find(c) >= 0:
        return i + 1, c

  return len(txt), ''


def split_into_sentences(txt, max_len_chars=150) -> []:
  spans = []
  begin = 0

  while begin < len(txt):
    segment = txt[begin:begin + max_len_chars]
    if len(segment) < max_len_chars:
      end = len(segment)
    else:
      end, __char = find_best_sentence_end(segment)
    span = begin, begin + end
    begin = span[1]
    spans.append(span)

  return spans


def is_long_enough(val: str, minlen=2) -> bool:
  if not val:
    return False
  if val.strip() == '':
    return False
  if len(val.strip()) < minlen:
    return False
  return True


def span_len(span: [int]) -> int:
  return abs(span[1] - span[0])


def _count_capitals(txt):
  s = 0
  for c in txt:
    if c.isupper():
      s += 1
  return s


def _count_digits(txt):
  s = 0
  for c in txt:
    if c.isdigit():
      s += 1
  return s


def compare_masked_strings(a, b, masked_substrings):
  a1 = a
  b1 = b
  for masked in masked_substrings:
    if a1.find(masked) >= 0 and b1.find(masked) >= 0:
      a1 = a1.replace(masked, '')
      b1 = b1.replace(masked, '')

  return jaro.get_jaro_distance(a1, b1, winkler=False, scaling=0.1)


def find_top_spans(paragraph_attention_vector, threshold=0.5, maxgap=2, limit=None) -> []:
  result = []
  top_indices = [i for i, v in enumerate(paragraph_attention_vector) if v > threshold]
  if len(top_indices) == 0:
    return result

  span_start = top_indices[0]
  averages = []
  i_prev = span_start

  for _p, i in enumerate(top_indices):
    if i - i_prev > maxgap:  # break
      sp = slice(span_start, i_prev + 1)
      result.append(sp)
      averages.append(np.mean(paragraph_attention_vector[sp]))
      span_start = i

    i_prev = i

  sp = slice(span_start, top_indices[-1] + 1)
  averages.append(np.mean(paragraph_attention_vector[sp]))
  result.append(sp)

  tops = np.argsort(averages, )[::-1]
  result = [result[i] for i in tops]
  if limit is not None:
    result = result[:limit]

  return result
