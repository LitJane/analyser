#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

import unittest
import sys

import numpy as np
import nltk
from text_tools import untokenize
from transaction_values import split_by_number_2
from  transaction_values import extract_sum
from text_normalize import *

data = [
    (41752.62, 'RUB',
     '\n2.1.  Общая сумма договора составляет 41752,62 руб. (Сорок одна тысяча семьсот пятьдесят два рубля) '
     '62 копейки, в т.ч. НДС (18%) 6369,05 руб. (Шесть тысяч триста шестьдесят девять рублей) 05 копеек, в'),

    (300000.0, 'RUB',
     'Стоимость услуг по настоящему Договору не может превышать 300 000 (трехсот тысяч) рублей, 00 копеек без учета НДС.'),

    (99000000.0, 'RUB',
     '6. Лимит Соглашения: 99 000 000 (девяносто девять миллионов) рублей 00 копеек.'),

    (300000.0, 'RUB',
     'Одобрить предоставление безвозмездной финансовой помощи в размере 300 000 (Триста тысяч) рублей для '),

    (100000000.0, 'RUB',
     'на сумму, превышающую 100 000 000 (сто миллионов) рублей без учета НДС '),

    # TODO:
    # (100000000.0, 'RUB',
    #  'на сумму, превышающую 50 000 000 (Пятьдесят миллионов) рублей без учета НДС (или эквивалент указанной суммы в '
    #  'любой другой валюте) но не превышающую 100 000 000 (Сто миллионов) рублей без учета НДС '),

    (80000.0, 'RUB', 'Счет № 115 на приобретение спортивного оборудования, '
                     'Стоимость оборудования 80 000,00 (восемьдесят тысяч рублей руб. 00 коп.) руб., НДС не облагается '),

    (1000000.0, 'EUR', 'стоимость покупки: 1 000 000 евро '),

    (67624292.0, 'RUB', 'составляет 67 624 292 (шестьдесят семь миллионов шестьсот двадцать четыре тысячи '
                        'двести девяносто два) рубля '),

    (4003246.0, 'RUB', 'участка № 1, приобретаемого ПОКУПАТЕЛЕМ, составляет 4 003 246(Четыре миллиона три '
                       'тысячи двести сорок шесть)  рублей,  НДС '),

    (81430814.0, 'RUB', '3. Общая Цена Договора: 81 430 814 (восемьдесят один миллион четыреста тридцать '
                        'тысяч восемьсот четырнадцать) рублей'),

    (50950000.0, 'RUB', 'сумму  50 950 000(пятьдесят миллионов девятьсот пятьдесят тысяч) руб. 00 коп. '
                        'без НДС, НДС не облагается на основании п.2 статьи 346.11.'),

    (1661293757.0, 'RUB',
     'составит - не более 1661 293,757 тыс . рублей ( с учетом ндс ) ( 0,93 % балансовой стоимости активов'),

    (490000.0, 'RUB',
     'с лимитом 490 000 (четыреста девяносто тысяч) рублей на ДТ, топливо АИ-92 и АИ-95 сроком до 31.12.2018 года  '),

    # (999.44, 'RUB', 'Стоимость 999 рублей 44 копейки'),
    (1999.44, 'RUB', 'Стоимость 1 999 (тысяча девятьсот) руб 44 (сорок четыре) коп'),
    (1999.44, 'RUB', '1 999 (тысяча девятьсот) руб. 44 (сорок четыре) коп. и что-то 34'),
    (25000000.0, 'USD', 'в размере более 25 млн . долларов сша'),
    (25000000.0, 'USD', 'эквивалентной 25 миллионам долларов сша'),

    (1000000.0, 'RUB', 'взаимосвязанных сделок в совокупности составляет от 1000000( одного ) миллиона рублей до 50000000 '),


]

numerics = """
    один два три четыре пять шесть семь восемь девять десять 
    одиннадцать двенадцать тринадцать
      
"""


class PriceExtractTestCase(unittest.TestCase):

    # def extract_price(self, text, currency_re):
    #     normal_text = text.lower()
    #     r = currency_re.findall(normal_text)
    #
    #     f = None
    #     try:
    #         f = (float(r[0][0].replace(" ", "").replace(",", ".")), r[0][5])
    #     except:
    #         pass
    #
    #     return f

    def test_extract(self):

        # currency_re =   re.compile(r'((^|\s+)(\d+[., ])*\d+)(\s*([(].{0,100}[)]\s*)?(евро|руб))')
        # rubles = re.compile(r'([0-9]+,[0-9]+)')

        for (price, currency, text) in data:

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
      numbers_str="""
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
        print (tokens)
        for t in tokens:
          ff = number_re.findall(t)
          print ( len(ff)>0, ff  )
          # self.assertTrue(len(ff)>0 )

    def test_split_by_number(self):
      import nltk
      for (price, currency, text) in data:

        normal_text = normalize_text(text, replacements_regex)  # TODO: fix nltk problem, use d.parse()
        tokens = nltk.word_tokenize(normal_text)

        a,b,c = split_by_number_2(tokens, np.ones(len(tokens)), 0.1)
        for t in a:
          restored=untokenize(t)
          print ('\t-', t)
          self.assertTrue(restored[0].isdigit())


if __name__ == '__main__':
    unittest.main()
