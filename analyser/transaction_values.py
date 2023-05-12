#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


# transaction_values.py

import math
import re
import warnings

from analyser.ml_tools import TokensWithAttention
from analyser.structures import currencly_map
from analyser.text_tools import to_float


class ValueConstraint:
  def __init__(self, value: float, currency: str, sign: int, context: TokensWithAttention):
    warnings.warn("ValueConstraint is deprecated, use TaggedValueConstraint", DeprecationWarning)
    assert context is not None

    self.value: float = value
    self.currency: str = currency
    self.sign: int = sign

    self.context: TokensWithAttention = context

  def __str__(self):
    return f'{self.value} {self.sign} {self.currency}'


complete_re = re.compile(
  # r'(свыше|превыша[а-я]{2,4}|не превыша[а-я]{2,4})?\s+'
  r'('
  r'(?P<digits>\d+([\.,\= ]\d+)*)'  # digits #0
  r'(?:\s*\(.+?\)\s*(?:тыс[а-я]*|млн|милли[а-я]{0,4})\.?)?'  # bullshit like 'от 1000000 ( одного ) миллиона рублей'
  r'(\s*(?P<qualifier>тыс[а-я]*|млн|милли[а-я]{0,4})\.?)?'  # *1000 qualifier
  r'(\s*\((?:(?!\)).)+?\))?\s*'  # some shit in parenthesis
  r'(?P<currency>руб[а-я]{0,4}|доллар[а-я]{1,2}|евро|тенге[\.,]?)'  # currency #7
  r'(\s*\((?:(?!\)).)+?\))?\s*'  # some shit in parenthesis
  r'((?P<cents>\d+)(\s*\(.+?\))?\s*коп[а-я]{0,4})?'  # cents
  r'(\s*.{1,5}(?P<vat>(учётом|учетом|включая|т\.ч\.|том числе)\s*(ндс|ндфл))(\s*\((?P<percent>\d{1,2})\%\))?)?'
  r')|('
  r'(?P<digits1>\d+([\., ]\d+)*)\s*\)\s*'
  r'(\s*(?P<qualifier1>тыс[а-я]*|млн|милли[а-я]{0,4})\.?)?'  # *1000 qualifier
  r'(?P<currency1>руб[а-я]{0,4}|доллар[а-я]{1,2}|евро|тенге[\.,]?)'
  r'.{0,25}?'
  r'\s*\(\s*((?P<cents1>\d+)\s*\)\s*коп[а-я]{0,4})?'
  r'(\s*.{1,5}(?P<vat1>(учётом|учетом|включая|т\.ч\.|том числе)\s*(ндс|ндфл))(\s*\(\s*(?P<percent1>\d+)\%\s*\))?)?'
  r')'
  ,
  re.MULTILINE | re.IGNORECASE
)

_re_greather_then_1 = re.compile(r'(не менее|не ниже)', re.MULTILINE)
_re_greather_then = re.compile(r'(\sот\s+|больше|более|свыше|выше|превыша[а-я]{2,4})', re.MULTILINE)
_re_less_then = re.compile(
  r'(до\s+|менее|не может превышать|лимит соглашения[:]*|не более|не выше|не превыша[а-я]{2,4})', re.MULTILINE)


def detect_sign(prefix: str):
  warnings.warn("use detect_sign_2", DeprecationWarning)
  a = _re_greather_then_1.findall(prefix)
  if len(a) > 0:
    return +1

  a = _re_less_then.findall(prefix)
  if len(a) > 0:
    return -1
  else:
    a = _re_greather_then.findall(prefix)
    if len(a) > 0:
      return +1
  return 0


number_re = re.compile(r'^\d+[,.]?\d+', re.MULTILINE)

VALUE_SIGN_MIN_TOKENS = 5


def find_value_spans(_sentence: str, vat_percent=0.20) -> ([int], float, [int], str, bool, float):
  for match in complete_re.finditer(_sentence):

    ix = ''
    if match['digits1'] is not None:
      ix = '1'

    # NUMBER
    number_span = match.span('digits' + ix)

    number: float = to_float(_sentence[number_span[0]:number_span[1]])

    # NUMBER MULTIPLIER
    qualifier_span = match.span('qualifier' + ix)
    qualifier = _sentence[qualifier_span[0]:qualifier_span[1]]
    if qualifier:
      if qualifier.startswith('тыс'):
        number *= 1e3
      else:
        if qualifier.startswith('м'):
          number *= 1e6

    # FRACTION (CENTS, KOPs)
    cents_span = match.span('cents' + ix)
    r_cents = _sentence[cents_span[0]:cents_span[1]]
    if r_cents:
      frac, whole = math.modf(number)
      if frac == 0:
        number += to_float(r_cents) / 100.

    # CURRENCY
    currency_span = match.span('currency' + ix)
    currency = _sentence[currency_span[0]:currency_span[1]]
    curr = currency[0:3]
    currencly_name = currencly_map[curr.lower()]

    original_sum = number

    vat_span = match.span('vat' + ix)
    r_vat = _sentence[vat_span[0]:vat_span[1]]
    including_vat = False
    if r_vat:

      vat_percent_span = match.span('percent' + ix)
      r_vat_percent = _sentence[vat_percent_span[0]:vat_percent_span[1]]
      if r_vat_percent:
        vat_percent = to_float(r_vat_percent) / 100

      number = number / (1. + vat_percent)
      # number = int(number * 100.) / 100.  # dumned truncate!
      number = round(number, 2)  # not truncate, round!
      including_vat = True

    # TODO: include fration span to the return value

    ret = number_span, number, currency_span, currencly_name, including_vat, original_sum, r_vat

    return ret


class ValueSpansFinder:
  def __init__(self, _sentence: str, vat_percent=0.20):
    self.number_span, self.value, self.currency_span, self.currencly_name, self.including_vat, self.original_sum, self.vat = find_value_spans(
      _sentence,
      vat_percent=vat_percent)

  def __str__(self):
    return f'{self.number_span}, {self.value}, {self.currency_span}, {self.currencly_name}, {self.including_vat}, {self.original_sum}, vat = {self.vat}'


if __name__ == '__main__':
  ex = '2.2 Общая стоимость Услуг составляет шестьдесят два миллиона (62000000) рублей ноль (30) копеек, включая НДС (20%): ' \
       'Десять миллионов четыреста тысяч (10400000) рубля ноль (00) копеек. Стоимость Услуг является фиксированной (твердой) ' \
       'и не подлежит изменению в течение срока действия Договора.'

  val = find_value_spans(ex)
  print('val', val)

  ex = "составит - не более 1661 293,757 тыс. рублей  25 копеек ( с учетом ндс ) ( 0,93 % балансовой стоимости активов)"
  val = find_value_spans(ex)
  print('val', val)
