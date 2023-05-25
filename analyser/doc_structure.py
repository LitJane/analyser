import re

from analyser.ml_tools import FixedVector
from analyser.text_tools import roman_might_be, Tokens, string_to_ip


def _count_start_whitespace(tokens: Tokens) -> int:
  for i, token in enumerate(tokens):

    if token != '.' and token != ' ' and token != '\t':
      return i
  return len(tokens)


def get_tokenized_line_number(tokens: Tokens, last_level):
  """	
  :param tokens: list of stings, supposed to be lowercase	
  :return: number, number_region, line???, level (for 1.1.1 it is 3)	
  """

  if len(tokens) == 0:
    return None, (0, 0), last_level + 1, False

  r = roman_might_be(tokens[0])

  if r:
    # Roman number
    return [r], (0, 1), 0, True

  else:
    # Arabic (might be)
    # Searching for numbered lines
    token_index = 0

    if tokens[0].lower() == 'статья' and len(tokens) >= 2:
      token_index = 1

    x = re.search(r'^(\d{1,2}($|\.)){1,6}', tokens[token_index], flags=re.MULTILINE)
    if x is not None:
      # found number
      n = string_to_ip(x.group(0))
      level = len(n)
      offset = token_index + 1  # tokens before actual text

      if len(tokens) > 2:
        if tokens[token_index + 1] == ')':
          level += 1
          offset += 1

        offset += _count_start_whitespace(tokens[offset:])

      return n, (0, offset), level, False

    else:
      # Number not found
      # searching for bulletpoints -
      x = re.search(r'^[\-|•]', tokens[token_index], flags=re.MULTILINE)
      if x is not None:
        return [-1], (0, 1), last_level, False
      else:
        pass

  return None, (0, 0), last_level, False


def remove_similar_indexes_considering_weights(indexes: [int], weights: FixedVector) -> [int]:
  hif = []

  def is_index_far(i):
    if i == 0:
      return True
    return indexes[i] - indexes[i - 1] > 1

  def is_bigger_confidence(i):
    id_ = indexes[i]
    id_p = hif[-1]
    return weights[id_] > weights[id_p]

  for i in range(len(indexes)):
    id_ = indexes[i]

    if is_index_far(i):
      hif.append(id_)
    elif is_bigger_confidence(i):
      # replace
      hif[-1] = id_

  return hif
