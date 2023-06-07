#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from analyser.doc_structure import remove_similar_indexes_considering_weights, get_tokenized_line_number


class DocumentStructureTestCase(unittest.TestCase):

  def test_remove_similar_indexes_considering_weights(self):
    a = []
    w = []

    r = remove_similar_indexes_considering_weights(a, w)
    self.assertEqual(r, [])

  def test_remove_similar_indexes_considering_weights_2(self):
    a = [1, 2]
    w = [99, 0, 1, 99]

    r = remove_similar_indexes_considering_weights(a, w)
    self.assertEqual(r, [2])

  def test_remove_similar_indexes_considering_weights_3(self):
    a = [1, 2]
    w = [99, 1, 0, 99]

    r = remove_similar_indexes_considering_weights(a, w)
    self.assertEqual(r, [1])

  def test_remove_similar_indexes_considering_weights_4(self):
    a = [1, 2, 4]
    w = [99, 1, 0, 99, 0]

    r = remove_similar_indexes_considering_weights(a, w)
    self.assertEqual(r, [1, 4])

  def test_remove_similar_indexes_considering_weights_5(self):
    a = [1, 2, 4, 5]
    w = [99, 1, 0, 99, 0, 1]

    r = remove_similar_indexes_considering_weights(a, w)
    self.assertEqual(r, [1, 5])

  def test_get_tokenized_line_number(self):
    n, _, level, roman = get_tokenized_line_number('2. correct'.split(' '), 0)
    self.assertEqual([2], n)
    self.assertEqual(1, level)
    self.assertEqual(False, roman)

    n, _, level, roman = get_tokenized_line_number('22. correct'.split(' '), 0)
    self.assertEqual([22], n)
    self.assertEqual(1, level)
    self.assertEqual(False, roman)

    n, _, level, roman = get_tokenized_line_number('II correct Roman II	'.split(' '), 0)
    self.assertEqual([2], n)
    self.assertEqual(0, level)
    self.assertEqual(True, roman)


if __name__ == '__main__':
  unittest.main()
