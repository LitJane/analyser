#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


# transaction_values.py

import math
import re
import warnings
from typing import List

from ml_tools import TokensWithAttention
from text_tools import to_float

currencly_map = {
  'руб': 'RUB',
  'дол': 'USD',
  'евр': 'EURO',
  'тэн': 'KZT',
  'тен': 'KZT',
}


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
  r'(?P<digits>\d+([., ]\d+)*)'  # digits #0
  r'(?:\s*\(.+?\)\s*(?:тыс[а-я]*|млн|милли[а-я]{0,4})\.?)?'  # bullshit like 'от 1000000 ( одного ) миллиона рублей'
  r'(\s*(?P<qualifier>тыс[а-я]*|млн|милли[а-я]{0,4})\.?)?'  # *1000 qualifier
  r'(\s*\((?:(?!\)).)+?\))?\s*'  # some shit in parenthesis 
  r'?((?P<currency>руб[а-я]{0,4}|доллар[а-я]{1,2}|евро|тенге)[\.,]?)'  # currency #7
  r'(\s*\((?:(?!\)).)+?\))?\s*'  # some shit in parenthesis 
  r'(\s*(?P<cents>\d+)(\s*\(.+?\))?\s*коп[а-я]{0,4})?'  # cents
  r'(\s*.{1,5}(?P<vat>(учётом|учетом|включая|т\.ч\.|том числе)\s*ндс)(\s*\((?P<percent>\d{1,2})\%\))?)?'
  ,
  re.MULTILINE | re.IGNORECASE
)


# for r in re.finditer(complete_re, text):
def extract_sum(_sentence: str, vat_percent=0.20) -> (float, str):
  warnings.warn("use find_value_spans", DeprecationWarning)
  r = complete_re.search(_sentence)

  if r is None:
    return None, None

  number = to_float(r[1])
  r_num = r[4]
  if r_num:
    if r_num.startswith('тыс'):
      number *= 1000
    else:
      if r_num.startswith('м'):
        number *= 1000000

  r_cents = r[10]
  if r_cents:
    frac, whole = math.modf(number)
    if frac == 0:
      number += to_float(r_cents) / 100.

  vat_span = r.span('vat')
  r_vat = _sentence[vat_span[0]:vat_span[1]]
  including_vat = False
  if r_vat:

    vat_percent_span = r.span('percent')
    r_vat_percent = _sentence[vat_percent_span[0]:vat_percent_span[1]]
    if r_vat_percent:
      vat_percent = to_float(r_vat_percent)/100
      # print(f'vat_percent::{vat_percent}')

    number = number / (1. + vat_percent)
    including_vat = True


  curr = r[7][0:3]

  return number, currencly_map[curr.lower()], including_vat


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

VALUE_SIGN_MIN_TOKENS = 4


def find_value_spans(_sentence: str, vat_percent=0.20) -> (List[int], float, List[int], str):
  for match in complete_re.finditer(_sentence):

    # NUMBER
    number_span = match.span('digits')

    number = to_float(_sentence[number_span[0]:number_span[1]])

    # NUMBER MULTIPLIER
    qualifier_span = match.span('qualifier')
    qualifier = _sentence[qualifier_span[0]:qualifier_span[1]]
    if qualifier:
      if qualifier.startswith('тыс'):
        number *= 1000
      else:
        if qualifier.startswith('м'):
          number *= 1000000

    # FRACTION (CENTS, KOPs)
    cents_span = match.span('cents')
    r_cents = _sentence[cents_span[0]:cents_span[1]]
    if r_cents:
      frac, whole = math.modf(number)
      if frac == 0:
        number += to_float(r_cents) / 100.

    # CURRENCY
    currency_span = match.span('currency')
    currency = _sentence[currency_span[0]:currency_span[1]]
    curr = currency[0:3]
    currencly_name = currencly_map[curr.lower()]

    vat_span = match.span('vat')
    r_vat = _sentence[vat_span[0]:vat_span[1]]
    including_vat = False
    if r_vat:

      vat_percent_span = match.span('percent')
      r_vat_percent = _sentence[vat_percent_span[0]:vat_percent_span[1]]
      if r_vat_percent:
        vat_percent = to_float(r_vat_percent) / 100
        # print(f'vat_percent::{vat_percent}')

      number = number / (1.+vat_percent)
      including_vat  = True

    # TODO: include fration span to the return value
    ret = number_span, number, currency_span, currencly_name #, including_vat

    return ret


if __name__ == '__main__':
  ex = "составит - не более 1661 293,757 тыс. рублей  25 копеек ( с учетом ндс ) ( 0,93 % балансовой стоимости активов)"
  val = find_value_spans(ex)
  print('extract_sum', extract_sum(ex))
  print('val', val)

  print(extract_sum('\n2.1.  Общая сумма договора составляет 41752 руб. (Сорок одна т'
                    'ысяча семьсот пятьдесят два рубля) '
                    '62 копейки, в т.ч. НДС (18%) 6369,05 руб. (Шесть тысяч триста шестьдесят девять рублей) 05 копеек, в'))

  print(extract_sum('взаимосвязанных сделок в совокупности составляет от '
                    '1000000 ( одного ) миллиона рублей до 50000000 '))


if __name__ == '__main__X':
  print(extract_sum('\n2.1.  Общая сумма договора составляет 41752 руб. (Сорок одна т'
                    'ысяча семьсот пятьдесят два рубля) '
                    '62 копейки, в т.ч. НДС (18%) 6369,05 руб. (Шесть тысяч триста шестьдесят девять рублей) 05 копеек, в'))
  print(extract_sum('эквивалентной 25 миллионам долларов сша'))

  print(extract_sum('взаимосвязанных сделок в совокупности составляет от '
                    '1000000 ( одного ) миллиона рублей до 50000000 '))
  print(extract_sum(
    'Общая сумма договора составляет 41752,62 рублей ( Сорок одна тысяча '
    'семьсот пятьдесят два рубля ) 62 копейки , в том числе НДС '))

  print(extract_sum("""
  одобрение заключения , изменения или расторжения какой-либо сделки общества , не указанной прямо в пункте 17.1 устава или настоящем пункте 22.5 ( за исключением крупной сделки в определении действующего законодательства российской федерации , которая подлежит одобрению общим собранием участников в соответствии с настоящим уставом или действующим законодательством российской федерации ) , если предметом такой сделки ( а ) является деятельность , покрываемая в долгосрочном плане , и сделка имеет стоимость , равную или превышающую 5000000 ( пять миллионов ) долларов сша , либо ( b ) является деятельность , не покрываемая в долгосрочном плане , и сделка имеет стоимость , равную или превышающую 500000 ( пятьсот тысяч ) долларов сша ;
  """))

  print(extract_sum(
    'одобрение заключения , изменения или расторжения какой-либо сделки общества , '
    'не указанной прямо в пункте 17.1 или настоящем пункте 22.5 ( за исключением '
    'крупной сделки в определении действующего законодательства российской федерации , '
    'которая подлежит одобрению общим собранием участников в соответствии с настоящим '
    'уставом или действующим законодательством российской федерации ) , если предметом '
    'такой сделки ( a ) является деятельность , покрываемая в долгосрочном плане , и '
    'сделка имеет стоимость , равную или превышающую 2000000 ( два миллиона ) долларов '
    'сша , но менее 5000000 ( пяти миллионов ) долларов сша , либо ( b ) деятельность , '
    'не покрываемая в долгосрочном плане , и сделка имеет стоимость , равную или '
    'превышающую 150000 ( сто пятьдесят тысяч ) долларов сша , но менее 500000 ( пятисот тысяч ) долларов сша ;'))

  print(extract_sum(
    '3. нецелевое расходование обществом денежных средств ( расходование не в соответствии'
    ' с утвержденным бизнес-планом или бюджетом ) при совокупности следующих условий : ( i ) размер '
    'таких нецелевых расходов в течение 6 ( шести ) месяцев превышает 25 000 000 ( двадцать пять миллионов ) рублей '
    'или эквивалент данной суммы в иной валюте , ( ii ) такие расходы не были в последующем одобрены советом директоров '
    'в течение 1 ( одного ) года с момента , когда нецелевые расходы превысили вышеуказанную сумму и ( iii ) генеральным директором '
    'не было получено предварительное одобрение на такое расходование от представителей участников , уполномоченных голосовать '
    'на общем собрании участников общества ;'))

  s_ = """Стоимость оборудования 80 000,00 ( восемьдесят тысяч рублей рублей 00 копеек ) рублей, НДС не облагается."""
  print(extract_sum(s_))

  s_ = """Стоимость оборудования 80 000,00 (восемьдесят тысяч рублей рублей 00 копеек) рублей,"""
  print(extract_sum(s_))
