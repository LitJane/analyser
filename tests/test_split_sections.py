#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

import numpy as np

from analyser.contract_parser import ContractDocument
from analyser.ml_tools import split_by_token_into_ranges, remove_similar_indexes


class SplitSectionsTest(unittest.TestCase):

  def test_split_by_token_into_ranges(self):
    tokens = ['1', '2', '3', '\n', '4', '5', '6', '\n', '7', '\n', '8']
    ranges = split_by_token_into_ranges(tokens, '\n')
    print(ranges)
    res = []
    for r in ranges:
      print(tokens[r])
      res.append(''.join(tokens[r]))

    self.assertEqual(len(res), 4)
    self.assertEqual(res[0], '123\n')
    self.assertEqual(res[1], '456\n')
    self.assertEqual(res[2], '7\n')
    self.assertEqual(res[3], '8')

  def test_split_by_token_into_ranges2(self):
    tokens = ['7', '\n']
    ranges = split_by_token_into_ranges(tokens, '\n')
    res = []
    for r in ranges:
      print(tokens[r])
      res.append(''.join(tokens[r]))

    self.assertEqual(len(res), 1)
    self.assertEqual(res[0], '7\n')

  def test_split_by_token_into_ranges3(self):
    tokens = ['7']
    ranges = split_by_token_into_ranges(tokens, '\n')
    res = []
    for r in ranges:
      print(tokens[r])
      res.append(''.join(tokens[r]))

    self.assertEqual(len(res), 1)
    self.assertEqual(res[0], '7')

  def test_find_sentence_beginnings(self):
    cd = ContractDocument('aaa\naaa\na')
    cd.parse()
    row_to_index = [6, 1]

    sentence_starts = cd.find_sentence_beginnings(row_to_index)
    print(sentence_starts)
    self.assertTrue(np.allclose(sentence_starts, [3, 0]))

  def test_find_sentence_beginnings_2(self):
    cd = ContractDocument('aaa\naaa\na')
    cd.parse()

    row_to_index = [6, 1]
    sentence_starts = cd.find_sentence_beginnings(row_to_index)
    print(sentence_starts)
    self.assertTrue(np.allclose(sentence_starts, [3, 0]))

  def test_remove_near_indexes_1(self):
    indexes = [0, 0, 0]
    filtered = remove_similar_indexes(indexes)
    print(filtered)
    self.assertTrue(np.allclose(filtered, [0]))

  def test_remove_near_indexes_2(self):
    indexes = [100, 10, 500]
    filtered = remove_similar_indexes(indexes, min_section_size=100)
    print(filtered)
    self.assertTrue(np.allclose(filtered, [100, 500]))


if __name__ == '__main__':
  unittest.main()
