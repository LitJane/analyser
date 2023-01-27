#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


# transaction_values.py

import re

from analyser.legal_docs import LegalDocument
from analyser.ml_tools import SemanticTag

case_number_re = re.compile(
  r'(дело|Дело|по\s*делу|дела)\s*(№|N|номер)\s*(?P<case_num>([А-Я]?\s?[0-9-/]*))', re.MULTILINE)


def find_case_number_spans(_sentence: str) -> ([int], float, [int], str, bool, float):
  for match in case_number_re.finditer(_sentence):
    # NUMBER
    number_span = match.span('case_num')
    if number_span is not None:
      number = _sentence[number_span[0]:number_span[1]]
      return number, (number_span[0], number_span[1])

  return None, None


def find_case_number(doc: LegalDocument, tagname='case_number') -> SemanticTag or None:
  _num, c_span = find_case_number_spans(doc.text)
  if c_span is None:
    return None
  span = doc.tokens_map.token_indices_by_char_range(c_span)
  return SemanticTag(tagname, _num, span)


if __name__ == '__main__':
  ex = '18 января 2022 года Дело № А 40-262624/2021-146-2012 45\nРезолютивная часть решения объявлена 13   2022 года'
  print('-', find_case_number_spans(ex))
  ex = '18 января 2022 по делу №40-262624/2021-146-2012 Резолютивная часть решения объявлена 13 января 2022  '
  print('-', find_case_number_spans(ex))
  ex = '18 января 2022 по делу N40-262624/2021-146-2012\nРезолютивная часть решения объявлена 13 января 2022  '
  print('-', find_case_number_spans(ex))
  ex = '18 января 2022 по делу номер 40-262624/2021-146-2012\nРезолютивная часть решения объявлена 13 января 2022 года'
  print('-', find_case_number_spans(ex))
  ex = 'мирового соглашения в рамках дела № А 56-113826/2019 о признании '
  print('-', find_case_number_spans(ex))
