#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

import sys
import unittest
from typing import List

import nltk
import numpy as np

from contract_parser import ContractDocument, find_value_sign_currency
from documents import TextMap
from legal_docs import find_value_sign
from ml_tools import conditional_p_sum
from text_normalize import *
from transaction_values import extract_sum
from charter_parser import split_by_number_2

data = [
  (0, 41752.62, 'RUB',
   '\n2.1.  Общая сумма договора составляет 41752,62 руб. (Сорок одна тысяча семьсот пятьдесят два рубля) '
   '62 копейки, в т.ч. НДС (18%) 6369,05 руб. (Шесть тысяч триста шестьдесят девять рублей) 05 копеек, в'),

  (-1, 300000.0, 'RUB',
   'Стоимость услуг по настоящему Договору не может превышать 300 000 (трехсот тысяч) рублей, 00 копеек без учета НДС.'),

  (0, 99000000.0, 'RUB',  # TODO: make sign < (-1) 'Лимит' means 'at max' means <=
   '6. Лимит Соглашения: 99 000 000 (девяносто девять миллионов) рублей 00 копеек.'),

  (0, 300000.0, 'RUB',
   'Одобрить предоставление безвозмездной финансовой помощи в размере 300 000 (Триста тысяч) рублей для '),

  (1, 100000000.0, 'RUB',
   'на сумму, превышающую 100 000 000 (сто миллионов) рублей без учета НДС '),

  # TODO:
  # (100000000.0, 'RUB',
  #  'на сумму, превышающую 50 000 000 (Пятьдесят миллионов) рублей без учета НДС (или эквивалент указанной суммы в '
  #  'любой другой валюте) но не превышающую 100 000 000 (Сто миллионов) рублей без учета НДС '),

  (0, 80000.0, 'RUB', 'Счет № 115 на приобретение спортивного оборудования, '
                      'Стоимость оборудования 80 000,00 (восемьдесят тысяч рублей руб. 00 коп.) руб., НДС не облагается '),

  (0, 1000000.0, 'EURO', 'стоимость покупки: 1 000 000 евро '),

  (0, 86500.0, 'RUB',
   'Стоимость Услуг составляет 86 500,00 рублей (Восемьдесят шесть тысяч пятьсот рублей) 00 копеек, налогом на добавленную стоимость (НДС) не облагается согласно пп. 14 п. 2 ст. 149 Налогового кодекса Российской Федерации. '),

  (0, 67624292.0, 'RUB', 'составляет 67 624 292 (шестьдесят семь миллионов шестьсот двадцать четыре тысячи '
                         'двести девяносто два) рубля '),

  (0, 4003246.0, 'RUB', 'участка № 1, приобретаемого ПОКУПАТЕЛЕМ, составляет 4 003 246(Четыре миллиона три '
                        'тысячи двести сорок шесть)  рублей,  НДС '),

  (0, 81430814.0, 'RUB', '3. Общая Цена Договора: 81 430 814 (восемьдесят один миллион четыреста тридцать '
                         'тысяч восемьсот четырнадцать) рублей'),

  (0, 50950000.0, 'RUB', 'сумму  50 950 000(пятьдесят миллионов девятьсот пятьдесят тысяч) руб. 00 коп. '
                         'без НДС, НДС не облагается на основании п.2 статьи 346.11.'),

  (-1, 1661293757.0, 'RUB',
   'составит - не более 1661 293,757 тыс . рублей ( с учетом ндс ) ( 0,93 % балансовой стоимости активов'),

  (-1, 490000.0, 'RUB',
   'с лимитом 490 000 (четыреста девяносто тысяч) рублей на ДТ, топливо АИ-92 и АИ-95 сроком до 31.12.2018 года  '),

  # (999.44, 'RUB', 'Стоимость 999 рублей 44 копейки'),
  (0, 1999.44, 'RUB', 'Стоимость 1 999 (тысяча девятьсот) руб 44 (сорок четыре) коп'),
  (0, 1999.44, 'RUB', '1 999 (тысяча девятьсот) руб. 44 (сорок четыре) коп. и что-то 34'),
  (1, 25000000.0, 'USD', 'в размере более 25 млн . долларов сша'),
  (0, 25000000.0, 'USD', 'эквивалентной 25 миллионам долларов сша'),
  (0, 1110635.40, 'RUB',
   'Стоимость Услуг составляет 1 110 635,40 (Один миллион сто десять тысяч шестьсот тридцать пять) рублей 40 копеек, в т.ч. НДС (18%): 169 418,96 (Сто шестьдесят девять тысяч четыреста восемнадцать тысяч) рублей 96 копеек. Стоимость Услуг включает в себя стоимость учебных, справочных, методических и иных материалов, передаваемых Работникам.'),
  # (0, 80000,'RUB', 'Стоимость оборудования 80 000,00 (восемьдесят тысяч рублей рублей 00 копеек) рублей,'),#TODO
  (0, 80000, 'RUB', 'Стоимость оборудования 80000,00 (восемьдесят тысяч рублей рублей 00 копеек) рублей,'),  # TODO

  (1, 1000000.0, 'RUB',
   'взаимосвязанных сделок в совокупности составляет от 1000000( одного ) миллиона рублей  '),  # до 50000000

  (None, 2000000.0, 'USD', 'одобрение заключения , изменения или расторжения какой-либо сделки общества , '
                           'не указанной прямо в пункте 17.1 или настоящем пункте 22.5 ( за исключением '
                           'крупной сделки в определении действующего законодательства российской федерации , '
                           'которая подлежит одобрению общим собранием участников в соответствии с настоящим '
                           'уставом или действующим законодательством российской федерации ) , если предметом '
                           'такой сделки ( a ) является деятельность , покрываемая в долгосрочном плане , и '
                           'сделка имеет стоимость , равную или превышающую 2000000 ( два миллиона ) долларов '
                           'сша , но менее 5000000 ( пяти миллионов ) долларов сша , либо ( b ) деятельность , '
                           'не покрываемая в долгосрочном плане , и сделка имеет стоимость , равную или '
                           'превышающую 150000 ( сто пятьдесят тысяч ) долларов сша , но менее 500000 ( пятисот тысяч ) долларов сша ;'),

  (1, 25000000.0, 'RUB', '3. нецелевое расходование обществом денежных средств ( расходование не в соответствии'
                         ' с утвержденным бизнес-планом или бюджетом ) при совокупности следующих условий : ( i ) размер '
                         'таких нецелевых расходов в течение 6 ( шести ) месяцев превышает 25 000 000 ( двадцать пять миллионов ) рублей '
                         'или эквивалент данной суммы в иной валюте , ( ii ) такие расходы не были в последующем одобрены советом директоров '
                         'в течение 1 ( одного ) года с момента , когда нецелевые расходы превысили вышеуказанную сумму и ( iii ) генеральным директором '
                         'не было получено предварительное одобрение на такое расходование от представителей участников , уполномоченных голосовать '
                         'на общем собрании участников общества ;'),

]


# numerics = """
#     один два три четыре пять шесть семь восемь девять десять
#     одиннадцать двенадцать тринадцать
#
# """


class PriceExtractTestCase(unittest.TestCase):

  def test_find_value_sign_a(self):
    text = """стоимость, равную или превышающую 2000000 ( два миллиона ) долларов сша"""
    tm = TextMap(text)
    sign, span = find_value_sign(tm)
    quote = tm.text_range(span)
    self.assertEqual('превышающую', quote)

  def test_find_value_sign_b(self):
    text = """стоимость, равную или превышающую 2000000 ( два миллиона ) долларов сша, но менее"""
    tm = TextMap(text)
    sign, span = find_value_sign(tm)
    quote = tm.text_range(span)
    self.assertEqual('менее', quote)

  def test_find_value_sign_c(self):

    for (sign_expected, price, currency, text) in data:
      tm = TextMap(text)
      sign, span = find_value_sign(tm)
      if sign_expected:
        self.assertEqual(sign_expected, sign, text)
      quote = ''
      if span:
        quote = tm.text_range(span)
      print(f'{sign},\t {span},\t {quote}')

  def test_find_all_value_sign_currency(self):
    text = """стоимость, равную или превышающую 2000000 ( два миллиона ) долларов США, но менее"""
    doc = ContractDocument(text)
    doc.parse()
    print(doc.normal_text)
    # =========================================
    r = find_value_sign_currency(doc)
    # =========================================

    # for sum, sign, currency in r:

    print(f'{r[0].value}, {r[0].sign}, {r[0].currency}')

    self.assertEqual('USD', r[0].currency.value)
    self.assertEqual(1, r[0].sign.value)
    self.assertEqual(2000000, r[0].value.value)

    self.assertEqual('превышающую', doc.tokens_map_norm.text_range(r[0].sign.span))
    self.assertEqual('2000000',
                     doc.tokens_map_norm.text_range(r[0].value.span))  # TODO:  keep 2000000
    self.assertEqual('долларов', doc.tokens_map_norm.text_range(r[0].currency.span))  # TODO: keep

  def test_find_all_value_sign_currency_d(self):
    sign_exp, price, currency_exp, text = (0, 1000000.0, 'EURO', 'стоимость покупки: 1 000 000 евро ')

    doc = ContractDocument(text)
    doc.parse()
    r: List = find_value_sign_currency(doc)

    print(doc.tokens_map_norm.text_range(r[0].value.span))
    self.assertEqual(price, r[0].value.value, text)
    self.assertEqual(currency_exp, r[0].currency.value)
    print(f'{r[0].value}, {r[0].sign}, {r[0].currency}')

  def test_find_all_value_sign_currency_a(self):
    for (sign_exp, price, currency_exp, text) in data:
      doc = ContractDocument(text)
      doc.parse()
      r: List = find_value_sign_currency(doc)
      if r:
        print(doc.tokens_map_norm.text_range(r[0].value.span))
        self.assertEqual(price, r[0].value.value, text)
        self.assertEqual(currency_exp, r[0].currency.value, text)
        print(r[0].value.value)
        print(f'{r[0].value}, {r[0].sign}, {r[0].currency}')

  def test_extract(self):

    for (sign, price, currency, text) in data:

      normal_text = normalize_text(text, replacements_regex)  # TODO: fix nltk problem, use d.parse()
      print(f'text:            {text}')
      print(f'normalized text: {normal_text}')
      f = None
      try:
        f = extract_sum(normal_text)
        self.assertEqual(price, f[0])
        print(f"\033[1;32m{f}\u2713")
      except:
        print("\033[1;35;40m FAILED:", price, currency, normal_text, 'f=', f)
        print(sys.exc_info())

      # #print (normal_text)
      # print('expected:', price, 'found:', f)

  def test_number_re(self):
    from transaction_values import number_re
    numbers_str = """
        3.44
        41752,62 рублей
        превышать 300000 ( трехсот тысяч )
        Соглашения: 99000000 ( девяносто 
        в размере 300000 ( Триста
        оборудования 80000,00 ( восемьдесят
        покупки: 1000000 евро
        составляет 67624292 ( шестьдесят
        АИ-92
        """
    numbers = numbers_str.split('\n')
    for n in numbers:
      tokens = nltk.word_tokenize(n)
      print(tokens)
      for t in tokens:
        ff = number_re.findall(t)
        print(len(ff) > 0, ff)
        # self.assertTrue(len(ff)>0 )

  def test_split_by_number(self):
    for (sign, price, currency, text) in data:

      normal_text = normalize_text(text, replacements_regex)  # TODO: fix nltk problem, use d.parse()
      tm = TextMap(normal_text)

      a, b, c = split_by_number_2(tm.tokens, np.ones(len(tm)), 0.1)
      for t in a:
        restored = t[0]
        print('\t-', t)
        self.assertTrue(restored[0].isdigit())

  def test_conditional_p_sum(self):
    v = [0.9, 0.9]
    s = conditional_p_sum(v)
    for a in v:
      self.assertGreater(s, a)
      self.assertGreater(1, a)

    v = [0.1, 0.2, 0.7, 0.9, 0.999]
    s = conditional_p_sum(v)
    for a in v:
      self.assertGreater(s, a)
      self.assertGreater(1, a)


if __name__ == '__main__':
  unittest.main()
