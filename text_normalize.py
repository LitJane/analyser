#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import re

from text_tools import unquote


def r_group(x, name=None):
  if name is not None:
    return f'(?P<{name}>{x})'
  else:
    return f'({x})'


r_quote_open = r_group(r'[«"<]\s?|[\'`]{2}\s?')
r_quote_close = r_group(r'\s?[»">]|\s?[\'`]{2}')


def ru_cap(xx):
  return '\s+'.join([f'[{x[0].upper()}{x[0].lower()}]{x[1:-2]}[а-я]{{0,3}}' for x in xx.split(' ')])


def r_bracketed(x, name=None):
  return r_group(r'[(]' + x + r'[)]', name)


r_few_words_s = r'\s+[А-Яа-я\-, ]{0,80}'

r_capitalized_ru = r'([А-Я][a-яА-Я–\-]{0,25})'
r_capitalized = r_group(r'[A-ZА-Я][a-zA-Za-яА-Я–\-]{0,25}')
# _r_name = r'[А-ЯA-Z][А-Яа-яA-Za-z\-–\[\]. ]{0,40}[а-яa-z.]'
# _r_name_ru_having_quote = r'«([А-Я][А-Яа-я\-–\[\].\s«]{0,40}[А-Яа-я.,])»'
_r_name_ru = r'[А-Я][№А-Яа-я\-–\[\].\s«]{0,40}[А-Яа-я.,]'
_r_name_lat = r'[A-Z][№A-Za-z\-–\[\].\s]{0,40}[A-Za-z,]'
_r_name = r_group(_r_name_ru) + '|' + r_group(_r_name_lat)
r_name = r_group(_r_name, 'name')

"""Puts name into qotes"""
r_human_name_part = r_capitalized

r_human_full_name = r_group(r_human_name_part + r'\s*' + r_human_name_part + '\s*' + r_human_name_part + '?\w')
r_human_abbr_name = r_group(r_human_name_part + r'\s*' + '([А-ЯA-Z][.]\s?){1,2}')
r_human_name = r_group(r_human_full_name + '|' + r_human_abbr_name, 'human_name')

"""Puts name into qotes"""


def r_quoted(x):
  assert x is not None
  return r_quote_open + r'\s*' + x + r'\s*' + r_quote_close


r_quoted_name = r_group(r_quoted(r_name), 'r_quoted_name')

spaces_regex = [
  (re.compile(r'\t'), ' '),
  (re.compile(r'[ ]{2}'), ' '),
  (re.compile(r' '), ' ')  # this is not just space char! this is weird invisible symbol

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
]

fixtures_regex = [
  (re.compile(r'(?<=[А-Я][)])\n'), '.\n'),
  (re.compile(r'(?<=[А-Я])\n'), '.\n'),

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

replacements_regex = table_of_contents_regex + dates_regex + abbreviation_regex + fixtures_regex + spaces_regex + syntax_regex + numbers_regex + formatting_regex


def normalize_text(_t: str, replacements_regex):
  t = _t.replace("''", r'"')
  for (reg, to) in replacements_regex:
    t = reg.sub(to, t)

  return t


_legal_entity_types_of_subsidiaries = ['АО', 'ООО', 'ТОО', 'ИООО', 'ЗАО', 'НИС а.о.']

r_quoted_name_contents = r_quote_open + r'\s*' + r_group(_r_name, 'r_quoted_name_contents') + r'\s*' + r_quote_close
r_quoted_name_contents_c = re.compile(r_quoted_name_contents)


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

  # x = r_quoted_name_contents_c.search(normal_name)
  # if x is not None and x['r_quoted_name_contents'] is not None:
  #   normal_name = x['r_quoted_name_contents']

  # normal_name = re.sub(r'["\']', '', normal_name)

  normal_name = unquote(normal_name)

  if normal_name.find('«') >= 0 and normal_name.find('»') < 0:  # TODO: hack
    normal_name += '»'

  return legal_entity_type, normal_name
