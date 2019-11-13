#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import warnings
from typing import List

import nltk
import numpy as np
import scipy.spatial.distance as distance

Tokens = List[str]


def find_ner_end(tokens, start, max_len=20):
  # TODO: use regex
  for i in range(start, len(tokens)):
    if tokens[i] == '"':
      return i

    elif tokens[i] == '»':
      return i

    elif tokens[i] == '\n':
      return i

    elif tokens[i] == '.':
      return i

    elif tokens[i] == ';':
      return i

  return min(len(tokens), start + max_len)


def to_float(string):
  try:
    return float(string.replace(" ", "").replace(",", "."))
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
    if d > 1: d = 1
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


"""

Kind of Moving Earth (or Fréchet distance)
ACHTUNG! This is not WMD

inspired by https://en.wikipedia.org/wiki/Earth_mover%27s_distance https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html

Compute matrix of pair-wize distances between words of 2 sentences (each 2 each)
For each word in sentence U, find the Distance to semantically nearest one in the other sentence V
The Sum of these minimal distances (or mean) is sort of effort required to transform U sentence to another, V sentence.
For balance (symmetry) swap U & V and find the effort required to strech V sentence to U.


"""


def dist_frechet_cosine_directed(u, v):
  d_ = distance.cdist(u, v, 'cosine')
  return d_.min(0).sum()


def dist_frechet_cosine_undirected(u, v):
  d1 = dist_frechet_cosine_directed(u, v)
  d2 = dist_frechet_cosine_directed(v, u)
  return round((d1 + d2) / 2, 2)


def dist_frechet_eucl_directed(u, v):
  d_ = distance.cdist(u, v, 'euclidean')
  return d_.min(0).sum()


def dist_frechet_eucl_undirected(u, v):
  d1 = dist_frechet_eucl_directed(u, v)
  d2 = dist_frechet_eucl_directed(v, u)
  return round((d1 + d2) / 2, 2)


def dist_mean_cosine_frechet(u, v):
  return dist_frechet_cosine_undirected(u, v) + dist_mean_cosine(u, v)


def dist_cosine_housedorff_directed(u, v):
  d_ = distance.cdist(u, v, 'cosine')
  return d_.min(0).max()


def dist_cosine_housedorff_undirected(u, v):
  d1 = dist_cosine_housedorff_directed(u, v)
  d2 = dist_cosine_housedorff_directed(v, u)
  return round((d1 + d2) / 2, 2)


# ----------------------------------------------------------------
# MISC
# ----------------------------------------------------------------


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


def tokenize_text(text):
  warnings.warn("deprecated, use TextMap(text)", DeprecationWarning)

  sentences = text.split('\n')
  result = []
  for i in range(len(sentences)):
    sentence = sentences[i]
    result += nltk.word_tokenize(sentence)
    if i < len(sentences) - 1:
      result += ['\n']

  return result


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


#
# def get_sentence_bounds_at_index(index, tokens):
#   warnings.warn("deprecated: method must be moved to TextMap class", DeprecationWarning)
#   start = find_token_before_index(tokens, index, '\n', 0)
#   end = find_token_after_index(tokens, index, '\n', len(tokens) - 1)
#   return start + 1, end


# def get_sentence_slices_at_index(index, tokens) -> slice:
#   warnings.warn("deprecated: method must be moved to TextMap class", DeprecationWarning)
#   start = find_token_before_index(tokens, index, '\n')
#   end = find_token_after_index(tokens, index, '\n')
#   if start < 0:
#     start = 0
#   if end < 0:
#     end = len(tokens)
#   return slice(start + 1, end)


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


def replace_tokens(tokens: Tokens, replacements_map):
  result = []
  for t in tokens:
    key = t.lower()
    if key in replacements_map:
      result.append(replacements_map[key])
    else:
      result.append(t)
  return result


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


def find_best_sentence_end(txt) -> int:
  delimiters_prio = ['\n', '.!?', ';', ',', "-—", ')', ':', ' ']

  for delimiters in delimiters_prio:

    for i in reversed(range(len(txt))):
      c = txt[i:i + 1]
      if delimiters.find(c) >= 0:
        return i + 1

  return len(txt)


def split_into_sentences(txt, max_len_chars=150):
  spans = []
  begin = 0
  while begin < len(txt):
    segment = txt[begin:begin + max_len_chars]
    end = find_best_sentence_end(segment)
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


if __name__ == '__main__':
  x = '12345 aaaa.1234 ttt. dfdfd. 0123456789'
  be = find_best_sentence_end(x)
  spans = split_into_sentences(x, max_len_chars=12)
  for span in spans:
    print('S >>>', x[span[0]:span[1]])

  # print('E >>>', x[be:])
