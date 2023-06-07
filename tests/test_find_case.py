#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from analyser.case_numbers import find_case_number_spans


class CaseNumberTestCase(unittest.TestCase):

  @unittest.skip
  def test_1_trailing_spaced_number(self):
    ex = '18 января 2022 года Дело № А 40-262624/2021-146-2012 45\nРезолютивная часть решения объявлена 13   2022 года'
    r = find_case_number_spans(ex)
    print('-', r)
    self.assertEqual(r[0], 'А 40-262624/2021-146-2012 45')

  def test_2_before_nl(self):
    ex = '18 января 2022 по делу N40-262624/2021-146-2012\nРезолютивная часть решения объявлена 13 января 2022  '
    r = find_case_number_spans(ex)
    print('-', r)
    self.assertEqual(r[0], '40-262624/2021-146-2012')

  def test_2(self):
    ex = '18 января 2022 по делу N40-262624/2021-146-2012 Резолютивная часть решения объявлена 13 января 2022  '
    r = find_case_number_spans(ex)
    print('-', r)
    self.assertEqual(r[0], '40-262624/2021-146-2012')

  def test_3(self):
    ex = 'мирового соглашения в рамках дела № А 56-113826/2019 о признании '
    r = find_case_number_spans(ex)
    print('-', r)
    self.assertEqual(r[0], 'А 56-113826/2019')

  def test_4(self):
    ex = '18 января 2022 по делу номер 40-262624/2021-146-2012\nРезолютивная часть решения объявлена 13 января 2022 года'
    r = find_case_number_spans(ex)
    print('-', r)
    self.assertEqual(r[0], '40-262624/2021-146-2012')


if __name__ == '__main__':
  unittest.main()
