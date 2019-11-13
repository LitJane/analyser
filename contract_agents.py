#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import re
import warnings
from typing import AnyStr, Match, Dict, List

from pyjarowinkler import distance

from gpn.gpn import subsidiaries
from hyperparams import HyperParameters
from legal_docs import LegalDocument
from ml_tools import SemanticTag
from structures import ORG_LEVELS_names
from text_normalize import r_group, r_bracketed, r_quoted, r_capitalized_ru, \
  _r_name, r_quoted_name, ru_cap, r_few_words_s, r_human_name, normalize_company_name
from text_tools import is_long_enough, span_len

_is_valid = is_long_enough

ORG_LEVELS_re = r_group('|'.join([ru_cap(x) for x in ORG_LEVELS_names]), 'org_structural_level') + r'\s'

ORG_TYPES_re = [
  ru_cap('Акционерное общество'), 'АО',
  ru_cap('Закрытое акционерное общество'), 'ЗАО',
  ru_cap('Открытое акционерное общество'), 'ОАО',
  ru_cap('Государственное автономное учреждение'),
  ru_cap('Муниципальное бюджетное учреждение'),
  ru_cap('Некоммерческое партнерство'),

  # ru_cap('учреждение'),

  ru_cap('Федеральное государственное унитарное предприятие'), 'ФГУП',
  ru_cap('Федеральное государственное бюджетное образовательное учреждение высшего образования'), 'ФГБОУ',
  ru_cap('образовательное учреждение высшего образования'),
  ru_cap('Федеральное казенное учреждение'),
  ru_cap('Частное учреждение дополнительного профессионального образования'), 'ЧУДПО',
  ru_cap('Частное учреждение'),
  ru_cap('Общественная организация'),
  ru_cap('Общество с ограниченной ответственностью'), 'ООО',
  ru_cap('Партнерство с ограниченной ответственностью'),
  ru_cap('Некоммерческая организация'),
  ru_cap('Автономная некоммерческая организация'), 'АНО',
  ru_cap('Благотворительный фонд'),
  ru_cap('Индивидуальный предприниматель'), 'ИП',

  r'[Фф]онд[а-я]{0,2}' + r_few_words_s,

]
_r_types_ = '|'.join([x for x in ORG_TYPES_re])

r_few_words = r'\s+[А-Я]{1}[а-я\-, ]{1,80}'

r_type_ext = r_group(r'[А-Яa-zа-яА-Я0-9\s]*', 'type_ext')
r_name_alias = r_group(_r_name, 'alias')

r_quoted_name_alias = r_group(r_quoted(r_name_alias), 'r_quoted_name_alias')
r_alias_prefix = r_group(''
                         + r_group(r'(именуе[а-я]{1,3}\s+)?в?\s*дал[а-я]{2,8}\s?[–\-]?') + '|'
                         + r_group(r'далее\s?[–\-]?\s?'), name='r_alias_prefix')
r_alias = r_group(r".{0,140}" + r_alias_prefix + r'\s*' + r_quoted_name_alias)

r_types = r_group(f'{_r_types_}', 'type') + r'\s'
r_type_and_name = r_types + r_type_ext + r_quoted_name

r_alter = r_group(r_bracketed(r'.{1,70}') + r'{0,2}', 'alt_name')
complete_re_str = r_type_and_name + r'\s*' + r_alter + r_alias + '?'
# ----------------------------------
complete_re = re.compile(complete_re_str, re.MULTILINE | re.IGNORECASE)

# ----------------------------------

entities_types = ['type', 'name', 'alt_name', 'alias', 'type_ext']


def clean_value(x: str) -> str or None:
  if x is None:
    return x
  return x.replace('\t', ' ').replace('\n', ' ').replace(' – ', '-').lower()


def _find_org_names(text: str) -> List[Dict]:
  warnings.warn("make semantic tags", DeprecationWarning)

  def _to_dict(m: Match[AnyStr]):
    warnings.warn("make semantic tags", DeprecationWarning)
    d = {}
    for entity_type in entities_types:
      d[entity_type] = (m[entity_type], m.span(entity_type))

    return d

  org_names = {}

  for r in re.finditer(complete_re, text):
    org = _to_dict(r)

    # filter similar out
    _name = normalize_company_name(org['name'][0])
    if _name not in org_names:
      org_names[_name] = org

  return list(org_names.values())


def find_org_names(doc: LegalDocument, max_names=2, tag_kind_prefix='') -> List[SemanticTag]:
  tags = []
  org_i = 0

  for m in re.finditer(complete_re, doc.text):
    org_i += 1

    if org_i <= max_names:
      for entity_type in entities_types:
        tagname = f'{tag_kind_prefix}org.{org_i}.{entity_type}'
        char_span = m.span(entity_type)

        # span = doc.tokens_map_norm.token_indices_by_char_range_2(char_span)
        # val = doc.tokens_map_norm.text_range(span)

        span = doc.tokens_map.token_indices_by_char_range_2(char_span)
        val = doc.tokens_map.text_range(span)
        confidence = 1.0 - (span[0] / len(doc))  # relative distance from the beginning of the document
        if span_len(char_span) > 1 and _is_valid(val):
          if 'name' == entity_type:
            legal_entity_type, val = normalize_company_name(val)
            known_org_name, best_similarity = find_closest_org_name(subsidiaries, val,
                                                                    HyperParameters.subsidiary_name_match_min_jaro_similarity)
            if known_org_name is not None:
              val = known_org_name['_id']
              confidence *= best_similarity

          tag = SemanticTag(tagname, val, span)
          tag.confidence = confidence
          if confidence > 0.2:
            tags.append(tag)
          else:
            msg = f"low confidence:{confidence} \t {entity_type} \t {span} \t{val} \t{doc.filename}"
            warnings.warn(msg)

        else:
          msg = f"invalid tag value: {entity_type} \t {span} \t{val} \t{doc.filename}"
          warnings.warn(msg)

  # fitering tags
  # ignore distant matches
  return tags


r_ip = r_group(r'(\s|^)' + ru_cap('Индивидуальный предприниматель') + r'\s*' + r'|(\s|^)ИП\s*', 'ip')
sub_ip_quoter = (re.compile(r_ip + r_human_name), r'\1«\g<human_name>»')
sub_org_name_quoter = (re.compile(r_quoted_name + r'\s*' + r_bracketed(r_types)), r'\g<type> «\g<name>» ')

sub_alias_quote = (re.compile(r_alias_prefix + r_group(r_capitalized_ru, '_alias')), r'\1«\g<_alias>»')
alias_quote_regex = [
  sub_alias_quote,
  sub_ip_quoter,
  sub_org_name_quoter
]


def compare_masked_strings(a, b, masked_substrings):
  a1 = a
  b1 = b
  for masked in masked_substrings:
    if a1.find(masked) >= 0 and b1.find(masked) >= 0:
      a1 = a1.replace(masked, '')
      b1 = b1.replace(masked, '')

  # print(a1, '--', b1)
  return distance.get_jaro_distance(a1, b1, winkler=True, scaling=0.1)


def find_closest_org_name(subsidiaries, pattern, threshold=HyperParameters.subsidiary_name_match_min_jaro_similarity):
  if pattern is None:
    return None, 0
  best_similarity = 0
  finding = None
  _entity_type, pn = normalize_company_name(pattern)

  for s in subsidiaries:
    for alias in s['aliases'] + [s['_id']]:
      similarity = compare_masked_strings(pn, alias, [])
      if similarity > best_similarity:
        best_similarity = similarity
        finding = s

  if best_similarity > threshold:
    return finding, best_similarity
  else:
    return None, best_similarity


if __name__ == '__main__':
  print(r_group(r_capitalized_ru, 'alias'))
  pass
