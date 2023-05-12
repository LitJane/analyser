#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import re

from analyser.text_tools import unquote


def r_group(x, name=None):
  if name is not None:
    return f'(?P<{name}>{x})'
  else:
    return f'({x})'


r_quote_open = r_group(r'[«"<“]\s?|[\'`]{2}\s?')
r_quote_close = r_group(r'\s?[»">”]|\s?[\'`]{2}')


def morphology_agnostic_re(x):
  if len(x) > 2:
    r = f'[{x[0].upper()}{x[0].lower()}]{x[1:-2]}[а-я]{{0,3}}'
    r = f'({r})'
    return r
  else:
    return f'({x})'


def ru_cap(xx):
  ret = r'\s+'.join(morphology_agnostic_re(x) for x in xx.split(' '))
  return ret  # // r'\s+'.join([f'[{x[0].upper()}{x[0].lower()}]{x[1:-2]}[а-я]{{0,3}}' for x in xx.split(' ')])+'|'+xx.upper()


def r_bracketed(x, name=None):
  return r_group(r'[(]' + x + r'[)]', name)


r_few_words_s = r'\s+[А-Яа-я\-, ]{0,80}'

r_capitalized_ru = r'([А-Я][a-яА-Я–\-]{0,25})'
r_capitalized = r_group(r'[A-ZА-Я][a-zA-Za-яА-Я–\-]{0,25}')
# _r_name = r'[А-ЯA-Z][А-Яа-яA-Za-z\-–\[\]. ]{0,40}[а-яa-z.]'
# _r_name_ru_having_quote = r'«([А-Я][А-Яа-я\-–\[\].\s«]{0,40}[А-Яа-я.,])»'
_r_name_ru = r'[А-Я][№А-Яа-я\-–\[\].\s«]{0,50}[А-Яа-я.,]'
_r_name_ru_with_number = r'[А-Я][0-9А-Яа-я\-–\[\].\s«]{0,85}[0-9А-Яа-я.,]'
_r_name_lat = r'[A-Z][№A-Za-z\-–\[\].\s]{0,50}[A-Za-z,]'
_r_name = r_group(_r_name_ru_with_number) + '|' + r_group(_r_name_lat)
r_name = r_group(_r_name, 'name')

"""Puts name into qotes"""
r_human_name_part = r_capitalized

r_human_name_part_norm = r_group(r'\b[А-ЯA-Z]\B[а-яa-z]{1,25}')
r_human_name_part_cap = r_group(r'[A-ZА-Я-]{0,25}')


r_human_full_name = r_group(r_human_name_part_norm + r'\s*' + r_human_name_part_norm + r'\s*' + r_human_name_part_norm + r'?\w')
r_human_abbr_name = r_group(r_human_name_part_norm + r'\s*' + r'([А-ЯA-Z][.]\s?){1,2}')
r_human_name = r_group(r_human_full_name + '|' + r_human_abbr_name, 'human_name')

r_human_name_compilled = re.compile(r'\W' + r_human_name, re.MULTILINE)

def r_quoted(x):
  if x is None:
    raise ValueError('provide x please')
  return r_quote_open + r'\s*' + x + r'\s*' + r_quote_close


r_quoted_name = r_group(r_quoted(r_name), 'r_quoted_name')

_bell = '\x07'
spaces_regex = [
  (re.compile(_bell), '\n'),
  (re.compile(r'\t'), ' '),
  (re.compile(r'[ ]{2}'), ' '),
  (re.compile(r' '), ' '),  # this is not just space char! this is weird invisible symbol

  (re.compile(r'№(?=\S)'), '№ ')
  # ,
  # (re.compile(r'\n{2}'), '\n')
]

abbreviation_regex = [
  (re.compile(r'(?<=\s)руб\.'), 'рублей'),
  (re.compile(r'(?<=\s)коп\.'), 'копеек'),

  (re.compile(r'(?<=\d)коп\.'), ' копеек'),
  (re.compile(r'(?<=\d)руб\.'), ' рублей'),

  (re.compile(r'(?<=\s)п\.?(?=\s?\d)'), 'пункт'),
  (re.compile(r'(?<=\s)д\.?(?=\s?\d)'), 'дом'),
  (re.compile(r'(?<=\s)ул\.?'), 'улица'),

  (re.compile(r'в\sт\.\s*ч\.'), 'в том числе'),

  (re.compile(r'(?<=\d{4})\s*г\.\s*\n'), ' год.\n'),
  (re.compile(r'(?<=\d{4})\s*г\.'), ' год'),

]

org_expand_regex = [
  (re.compile(r'(?<=\s)*ООО(?=\s+)'), 'Общество с ограниченной ответственностью'),
  (re.compile(r'^АО(?=\s+)'), 'Акционерное Общество'),
  (re.compile(r'(?<=\s)АО(?=\s+)'), 'Акционерное Общество'),
  (re.compile(r'(?<=\s)*ИП(?=\s+)'), 'Индивидуальный предприниматель'),
  (re.compile(r'(?<=\s)*ЗАО(?=\s+)'), 'Закрытое Акционерное Общество'),
]

syntax_regex = [
  (re.compile(r'(?<=[а-яА-Я])\.(?=[а-яА-Я])'), '. '),

  (re.compile(r'(?<=[а-яА-Я])(?=\d)'), ' '),

  (re.compile(r'(?<=[^0-9 ])\.(?=[\w])'), '. '),

  (re.compile(r'(?<=\s)\.(?=\d+)'), '0.'),
  (re.compile(r'(?<=\S)*\s+\.(?=\s+)'), '.'),
  (re.compile(r'(?<=\S)*\s+,(?=\s+)'), ','),

  (re.compile(r'(?<=\d)+г\.'), ' г.'),
  (re.compile(r'(?<=[ ])г\.(?=\S+)'), 'г. '),

  (re.compile(r'«\s'), '«'),
  (re.compile(r'\s»'), '»'),

  # (re.compile(r"[']{2}"), '\"'),
  # (re.compile(r"``"), '"')

]

cleanup_regex = [

  (re.compile(r'[«\"\']Стороны[»\"\']'), 'Стороны'),
  (re.compile(r'[«\"\']Сторона[»\"\']'), 'Сторона'),

  (re.compile(r'с одной стороны и\s*\n'), 'с одной стороны и '),

  # (re.compile(r'\n\s*(\d{1,2}[\.|)]?\.?\s?)+'), '.\n — '),  # remove paragraph numbers
  (re.compile(r'\.\s*\.'), '.')

]

dates_regex = [
  (re.compile(r'(\d{2})\.(\d{2})\.(\d{4})'), r'\1-\2-\3')
]

numbers_regex = [
  (re.compile(r'(?<=\d)+[. ](?=\d{3}\s*[(].{3,40}\sтысяч?)'), ''),  # 3.000 (Три тысячи)
  (re.compile(r'(?<=\d)+[. ](?=\d{3})[. ]?(?=\d{3})'), ''),  # 3.000 (Три тысячи)

  (re.compile(r'(?<=\d\.)([а-яa-z]{2,30})', re.IGNORECASE | re.UNICODE), r' \1'),  ##space after dot
]

fixtures_regex = [

  (re.compile(r'\(в том числе нескольких взаимосвязанных сделок\)'), ''),
  (re.compile(r'\(или эквивалент указанной суммы в любой другой валюте\)'), ''),
  (re.compile(r'\(или эквивалента указанной суммы в любой другой валюте\)'), ''),
  (re.compile(r'аренду/субаренду'), 'аренду / субаренду'),
  (re.compile(r'арендой/субарендой'), 'арендой / субарендой'),

  (re.compile(r'(?<=[А-Я][)])\n'), '.\n'),
  (re.compile(r'(?<=[А-Я])\n'), '.\n'),
  (re.compile(r'(У\sС\sТ\sА\sВ)', re.IGNORECASE | re.MULTILINE), 'УСТАВ'),

  (re.compile(r'FORMTEXT'), ''),
  (re.compile(r''), ' ')  # ACHTUNG!! this is not just a space

]

formatting_regex = [
  # (re.compile(r'\n\s{2,5}'), ' ')
  (re.compile(r'(?<=\d)+[(]'), ' (')
]

tables_regex = [
  #     (re.compile(r'\|'), ' '),
]

table_of_contents_regex = [
  (re.compile(r'(содержание|оглавление)\s+(^.+?$)?\s+'
              r'((^\s*(статья\s+)?\d{1,3}\.?\s*(.+?|(.+$\s*^.+?))\s+\d{1,5}$)\s*(^\.*?$))+',
              re.IGNORECASE | re.MULTILINE),
   ''),

]


def normalize_text(_t: str, replacements_regex_):
  t = _t.replace("''", r'"')
  for (reg, to) in replacements_regex_:
    t = reg.sub(to, t)

  return t


_legal_entity_types_of_subsidiaries = ['АО', 'ООО', 'ТОО', 'ИООО', 'ЗАО', 'НИС а.о.']

r_quoted_name_contents = r_quote_open + r'\s*' + r_group(_r_name, 'r_quoted_name_contents') + r'\s*' + r_quote_close
r_quoted_name_contents_c = re.compile(r_quoted_name_contents)


def fix_ru_quotes(s):
  if s is None:
    return None

  s = s.lstrip('«').rstrip('»')
  s = unquote(s)
  if s.find('«') >= 0 and s.find('»') < 0:  # TODO: hack
    s += '»'

  if s.find('»') >= 0 and s.find('«') < 0:  # TODO: hack
    s = '«' + s

  findings = r_quoted_name_contents_c.finditer(s)

  rr = re.sub(r'[«»]', ' ', s)
  for match in findings:
    span = match.span()
    rr = rr[:span[0]] + s[span[0]:span[1]] + rr[span[1]:]

  return rr.strip()


def normalize_company_name(name: str) -> (str, str):
  legal_entity_type = ''
  normal_name = name
  for c in _legal_entity_types_of_subsidiaries:
    if name.strip().startswith(c):
      legal_entity_type = c
      normal_name = name[len(c):]

  if '' == normal_name:
    normal_name = legal_entity_type
    legal_entity_type = ''

  normal_name = normal_name.replace('\t', ' ').replace('\n', ' ')
  normal_name = normal_name.strip()
  normal_name = re.sub(r'\s+', ' ', normal_name)
  normal_name = re.sub(r'[\s ]*[-–][\s ]*', '-', normal_name)

  normal_name = fix_ru_quotes(normal_name)

  return legal_entity_type, normal_name


ORG_TYPES_re = [
  ru_cap('Публичное акционерное общество'), 'ПАО',
  ru_cap('Акционерное общество'), 'АО',

  ru_cap('Закрытое акционерное общество'), 'ЗАО',
  ru_cap('Открытое акционерное общество'), 'ОАО',
  ru_cap('Государственное автономное учреждение'),
  ru_cap('Муниципальное бюджетное учреждение'),
  ru_cap('Некоммерческое партнерство'),

  # ru_cap('учреждение'),

  ru_cap('Федеральное государственное унитарное предприятие'), 'ФГУП',
  ru_cap('Федеральное государственное бюджетное образовательное учреждение высшего образования'), 'ФГБОУ',
  ru_cap('Федеральное государственное бюджетное учреждение'), 'ФГБУ',
  ru_cap('образовательное учреждение высшего образования'),
  ru_cap('Федеральное казенное учреждение'),
  ru_cap('Частное учреждение дополнительного профессионального образования'), 'ЧУДПО',
  ru_cap('Частное образовательное учреждение'), 'ЧОУ',
  ru_cap('Частное учреждение'),
  ru_cap('Общественная организация'),
  ru_cap('Общество с ограниченной ответственностью'), 'ООО',
  ru_cap('Партнерство с ограниченной ответственностью'),
  ru_cap('Некоммерческая организация'),
  ru_cap('Автономная некоммерческая организация'), 'АНО',
  ru_cap('Благотворительный фонд'),
  ru_cap('Индивидуальный предприниматель'), 'ИП',

  r'[Фф]онд[а-я]{0,2}' + r_few_words_s, 'Фонд[уоме]{0,3}'

]
_r_types_ = '|'.join([x for x in ORG_TYPES_re])
r_alias_prefix = r_group(''
                         + r_group(r'(именуе[а-я]{1,3}\s+)?в?\s*дал[а-я]{2,8}\s?[–\-]?') + '|'
                         + r_group(r'далее\s?[–\-]?(именуе[а-я]{1,3}\s+)?\s?'), name='r_alias_prefix')
r_types = r_group(f'{_r_types_}', 'type') + r'\s'
r_ip = r_group(r'(\s|^)' + ru_cap('Индивидуальный предприниматель') + r'\s*' + r'|(\s|^)ИП\s*', 'ip')
sub_ip_quoter = (re.compile(r_ip + r_human_name), r'\1«\g<human_name>»')
sub_org_name_quoter = (re.compile(r_quoted_name + r'\s*' + r_bracketed(r_types)), r'\g<type> «\g<name>» ')
sub_alias_quote = (re.compile(r_alias_prefix + r_group(r_capitalized_ru, '_alias')), r'\g<r_alias_prefix>«\g<_alias>»')

# add comma before именуемое
sub_alias_comma = (
  re.compile(
    r_group(r'.[»)]\s', '_pref') +
    r_group(r_alias_prefix + r_capitalized_ru, '_alias'),
    re.UNICODE | re.IGNORECASE),
  r'\g<_pref>, \g<_alias>'
)

alias_quote_regex = [
  sub_alias_comma,
  sub_alias_quote,
  sub_ip_quoter,
  sub_org_name_quoter
]
replacements_regex = alias_quote_regex + table_of_contents_regex + dates_regex + abbreviation_regex + fixtures_regex + spaces_regex + syntax_regex + numbers_regex + formatting_regex
