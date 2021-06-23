#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import re
import warnings

from pyjarowinkler import distance

from analyser.hyperparams import HyperParameters
from analyser.legal_docs import LegalDocument
from analyser.ml_tools import SemanticTag, put_if_better
from analyser.schemas import OrgItem
from analyser.structures import ORG_LEVELS_names, legal_entity_types
from analyser.text_normalize import r_group, r_bracketed, r_quoted, r_capitalized_ru, \
  _r_name, r_quoted_name, ru_cap, normalize_company_name, r_alias_prefix, r_types, r_human_name, morphology_agnostic_re
from analyser.text_tools import is_long_enough, span_len, compare_masked_strings
from gpn.gpn import subsidiaries

r_being_a_citizen = r_group(r'являющ[а-я]{2,5}\s*граждан[а-я]{2,5}', 'citizen')
r_being_a_human_citizen = r_group(r_human_name + r',*\s*' + r_being_a_citizen, 'human_citizen')


def re_legal_entity_type(xx):
  _all = r'\s+'.join(morphology_agnostic_re(x) for x in xx.split(' '))
  return f'(?P<org_type>{_all})'


legal_entity_types_re = {}

for __t in sorted(legal_entity_types, key=lambda x: -len(x)):
  _regex = re_legal_entity_type(__t)
  rr = re.compile(_regex, re.IGNORECASE | re.UNICODE)
  legal_entity_types_re[rr] = __t
  if __t != rr.match(__t)[0]:
    raise RuntimeError(f"{__t}")

_is_valid = is_long_enough

ORG_LEVELS_re = r_group('|'.join([ru_cap(x) for x in ORG_LEVELS_names]), 'org_structural_level') + r'\s'

r_few_words = r'\s+[А-Я]{1}[а-я\-, ]{1,80}'

r_type_ext = r_group(r'[А-Яa-zа-яА-Я0-9\s]*', 'type_ext')
r_name_alias = r_group(_r_name, 'alias')

r_quoted_name_alias = r_group(r_quoted(r_name_alias), 'r_quoted_name_alias')
r_alias = r_group(r".{0,140}?" + r_alias_prefix + r'\s*' + r_quoted_name_alias, '_alias_ext')

r_type_and_name = r_types + r_type_ext + r_quoted_name

r_alter = r_group(r_bracketed(r'.{1,70}?') + r'{0,2}', 'alt_name')
complete_re_str_org = r_type_and_name + r'\s*' + r_alter
complete_re_str = r_group(complete_re_str_org + "|" + r_being_a_human_citizen) + r_alias + '?'

# ----------------------------------
complete_re = re.compile(complete_re_str, re.MULTILINE | re.UNICODE | re.DOTALL)
complete_re_ignore_case = re.compile(complete_re_str, re.MULTILINE | re.UNICODE | re.IGNORECASE | re.DOTALL)

protocol_caption_complete_re = re.compile(complete_re_str_org, re.MULTILINE | re.UNICODE | re.DOTALL)
protocol_caption_complete_re_ignore_case = re.compile(complete_re_str_org,
                                                      re.MULTILINE | re.UNICODE | re.IGNORECASE | re.DOTALL)

# ----------------------------------

org_pieces = ['type', 'name', 'human_name', 'alt_name', 'alias', 'type_ext']


class ContractAgent(OrgItem):
  # org_pieces = ['type', 'name', 'alt_name', 'alias', 'type_ext']
  def __init__(self):

    super().__init__()
    self.human_name: SemanticTag or None = None
    self.alt_name: SemanticTag or None = None
    self.type_ext: SemanticTag or None = None

  def as_list(self):
    warnings.warn("use OrgItem", DeprecationWarning)
    return [self.__dict__[key] for key in org_pieces if self.__dict__[key] is not None]

  def confidence(self):
    confidence = 0
    for key in ['type', 'name', 'alias']:
      tag = self.__dict__[key]
      if tag is not None:
        confidence += tag.confidence

    return confidence / 3.0

  def is_valid(self):
    for child in self.as_list():
      if child is not None:
        return True
    return False


def clean_value(x: str) -> str or None:
  if x is None:
    return x
  return x.replace('\t', ' ').replace('\n', ' ').replace(' – ', '-').lower()


def find_org_names(doc: LegalDocument,
                   max_names=2,
                   tag_kind_prefix='',
                   parent=None,
                   decay_confidence=True,
                   audit_subsidiary_name=None, regex=complete_re,
                   re_ignore_case=complete_re_ignore_case) -> [SemanticTag]:
  warnings.warn("deprecated because it calls _rename_org_tags", DeprecationWarning)
  _all: [ContractAgent] = find_org_names_raw(doc, max_names, parent, decay_confidence, regex=regex,
                                             re_ignore_case=re_ignore_case)

  if audit_subsidiary_name:
    _all = sorted(_all, key=lambda a: a.name.value != audit_subsidiary_name)
  else:
    _all = sorted(_all, key=lambda a: a.name.value)

  return _rename_org_tags(_all, prefix=tag_kind_prefix, start_from=1)


def _rename_org_tags(all_: [ContractAgent], prefix='', start_from=1) -> [SemanticTag]:
  warnings.warn("please switch to attributes_tree struktur", DeprecationWarning)
  tags = []
  for group, agent in enumerate(all_):
    for tag in agent.as_list():
      tagname = f'{prefix}org-{group + start_from}-{tag.kind}'
      tag.kind = tagname
      tags.append(tag)

  return tags


def find_org_names_raw(doc: LegalDocument, max_names=2, parent=None, decay_confidence=True, regex=complete_re,
                       re_ignore_case=complete_re_ignore_case) -> [ContractAgent]:
  all_org_names = find_org_names_raw_by_re(doc,
                                           regex=regex,
                                           confidence_base=1,
                                           parent=parent,
                                           decay_confidence=decay_confidence)

  # if len(all) < 200:
  # falling back to case-agnostic regexp
  all_org_names += find_org_names_raw_by_re(doc,
                                            regex=re_ignore_case,  # case-agnostic
                                            confidence_base=0.75,
                                            parent=parent,
                                            decay_confidence=decay_confidence)

  # filter, keep unique names
  _map = {}
  for ca in all_org_names:
    if ca.name is not None:
      if ca.confidence() > 0.2:
        put_if_better(_map, ca.name.value, ca, lambda a, b: a.confidence() > b.confidence())

  res = list(_map.values())
  res = sorted(res, key=lambda a: -a.confidence())
  return res[:max_names]
  #


def find_org_names_raw_by_re(doc: LegalDocument, regex, confidence_base: float, parent=None,
                             decay_confidence=True) -> [ContractAgent]:
  all_: [ContractAgent] = []

  iter = [m for m in re.finditer(regex, doc.text)]

  for m in iter:
    ca = ContractAgent()
    all_.append(ca)
    for re_kind in org_pieces:  # like 'type', 'name', 'human_name', 'alt_name', 'alias' ...
      try:
        char_span = m.span(re_kind)
        if span_len(char_span) > 1:
          span = doc.tokens_map.token_indices_by_char_range(char_span)
          confidence = confidence_base
          if decay_confidence:
            confidence *= (1.0 - (span[0] / len(doc)))

          kind = re_kind
          if re_kind == 'human_name':
            kind = 'name'

          val = doc.tokens_map.text_range(span)
          val = val.strip()
          if _is_valid(val):
            tag = SemanticTag(kind, val, span, parent=parent)
            tag.confidence = confidence
            tag.offset(doc.start)
            ca.__dict__[kind] = tag
      except IndexError:
        pass

  # normalize org_name names by find_closest_org_name

  return all_


def normalize_contract_agent(ca: ContractAgent or OrgItem) -> OrgItem:
  if ca.name is not None:
    _, val = normalize_company_name(ca.name.value)
    ca.name.value = val
    known_org_name, best_similarity = find_closest_org_name(subsidiaries, val,
                                                            HyperParameters.subsidiary_name_match_min_jaro_similarity)
    if known_org_name is not None:
      ca.name.value = known_org_name['_id']
      ca.name.confidence *= best_similarity

  # normalize org_type names by find_closest_org_name
  if ca.type is not None:
    long_, _, confidence_ = normalize_legal_entity_type(ca.type.value)
    ca.type.value = long_
    ca.type.confidence *= confidence_

  return ca


def find_closest_org_name(subsidiaries_: [dict], pattern: str,
                          threshold=HyperParameters.subsidiary_name_match_min_jaro_similarity):
  if pattern is None:
    return None, 0
  best_similarity = 0
  finding = None
  _entity_type, pn = normalize_company_name(pattern)

  for s in subsidiaries_:
    for alias in s['aliases'] + [s['_id']]:
      similarity = compare_masked_strings(pn, alias, [])
      if similarity > best_similarity:
        best_similarity = similarity
        finding = s

  if best_similarity > threshold:
    return finding, best_similarity
  else:
    return None, best_similarity


def find_known_legal_entity_type(txt) -> [(str, str)]:
  stripped = txt.strip()
  if stripped in legal_entity_types:
    return [(stripped, legal_entity_types[stripped])]

  for t in legal_entity_types:
    if stripped == legal_entity_types[t]:
      return [(t, stripped)]

  found = []
  for r in legal_entity_types_re:

    match = r.match(stripped)
    if (match):
      normalized = legal_entity_types_re[r]
      found.append((normalized, legal_entity_types[normalized]))
  return found


def normalize_legal_entity_type(txt) -> (str, str, float):
  knowns = find_known_legal_entity_type(txt.strip())
  if len(knowns) > 0:
    if len(knowns) == 1:
      k = knowns[0]
      return k[0], k[1], distance.get_jaro_distance(k[0], txt, winkler=True, scaling=0.1)
    else:
      finding = '', '', 0
      for k in knowns:
        d = distance.get_jaro_distance(k[0], txt, winkler=True, scaling=0.1)
        if d > finding[2]:
          finding = k[0], k[1], d
      return finding
  else:
    return txt, '', 0.5


if __name__ == '__main__':
  print(r_group(r_capitalized_ru, 'alias'))
  pass
