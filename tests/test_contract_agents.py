#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from analyser.contract_agents import find_closest_org_name, ContractAgent
from analyser.contract_parser import check_org_intersections
from analyser.hyperparams import HyperParameters
from analyser.ml_tools import SemanticTagBase
from analyser.schemas import OrgItem
from analyser.text_tools import compare_masked_strings
from gpn.gpn import subsidiaries


class ContractAgentsTestCase(unittest.TestCase):

  def test_check_org_intersections_solo_org(self):

    ca1 = OrgItem()
    ca1.alias = SemanticTagBase()
    ca1.alias.span = [20, 30]
    ca1.alias.confidence = 0.9

    check_org_intersections([ca1])

    self.assertIsNotNone(ca1.alias)

  def test_check_org_intersections(self):

    ca1 = OrgItem()
    ca1.alias = SemanticTagBase()
    ca1.alias.span = [20, 30]
    ca1.alias.confidence = 0.9

    ca2 = OrgItem()
    ca2.alias = SemanticTagBase()
    ca2.alias.span = [25, 30]
    ca2.alias.confidence = 0.95  # preferred

    check_org_intersections([ca1, ca2])
    self.assertIsNone(ca1.alias)
    self.assertIsNotNone(ca2.alias)

  def test_check_org_intersections2(self):

    ca1 = OrgItem()
    ca1.alias = SemanticTagBase()
    ca1.alias.span = [20, 30]
    ca1.alias.confidence = 0.91

    ca2 = OrgItem()
    ca2.alias = SemanticTagBase()
    ca2.alias.span = [25, 30]
    ca2.alias.confidence = 0.90  # preferred

    check_org_intersections([ca1, ca2])
    self.assertIsNone(ca2.alias)
    self.assertIsNotNone(ca1.alias)

  def test_compare_masked_strings(self):
    similarity = compare_masked_strings('Газпром нефть-Мобильная карта', 'Газпромнефть-Мобильная карта', [])
    print('similarity=', similarity)
    self.assertGreater(similarity, HyperParameters.subsidiary_name_match_min_jaro_similarity)

  def test_find_closest_org_name(self):

    mc = {
      "_id": "Газпромнефть-Мобильная карта",
      "legal_entity_type": "АО",
      "aliases": [
        "Газпромнефть-Мобильная карта"
      ]
    }
    known_org_name, _ = find_closest_org_name([mc], 'Газпромнефть - Мбильная карта',
                                              HyperParameters.subsidiary_name_match_min_jaro_similarity)
    self.assertIsNotNone(known_org_name)

    known_org_name, _ = find_closest_org_name(subsidiaries, 'Газпромнефть - Мбильная карта',
                                              HyperParameters.subsidiary_name_match_min_jaro_similarity)
    self.assertIsNotNone(known_org_name)
    self.assertEqual(mc['_id'], known_org_name['_id'])

  def test_compare_masked_strings_1(self):
    s = compare_masked_strings('Многофункциональный комплекс «Лахта центр»',
                               'Многофункциональный комплекс «Лахта центр»', [])
    print(s)
    for s1 in subsidiaries:
      for name in s1['aliases']:
        s = compare_masked_strings(name, name, [])
        self.assertEqual(s, 1)

  def test_find_closest_org_name_solo(self):
    expected = 'Многофункциональный комплекс «Лахта центр»'
    s = find_closest_org_name(subsidiaries, 'Многофункциональный комплекс «Лахта центр»', 0)[0]
    self.assertEqual(expected, s['_id'])

    s = find_closest_org_name(subsidiaries, 'Многофункциональный комплекс Лахта центр', 0)[0]
    self.assertEqual(expected, s['_id'])

    s = find_closest_org_name(subsidiaries, 'Многофункциональный комплекс Лахта центр  ', 0)[0]
    self.assertEqual(expected, s['_id'])

    s = find_closest_org_name(subsidiaries, 'Многофункциональный комплекс Лахта-центр  ', 0)[0]

    s = find_closest_org_name(subsidiaries, 'Многофункциональный комплекс Лахта-центр'.upper(), 0)[0]

    s = find_closest_org_name(subsidiaries, 'Многофункциональный комплекс Лахта Центр', 0)[0]
    self.assertEqual(expected, s['_id'])

  def test_find_closest_org_name_solo2(self):
    expected = 'Газпромнефть-МНПЗ'
    s = find_closest_org_name(subsidiaries, 'Газпромнефть - МНПЗ', 0)[0]
    self.assertEqual(expected, s['_id'])

    expected = 'Газпромнефть-ОНПЗ'
    s = find_closest_org_name(subsidiaries, 'Газпромнефть-ОНПЗ', 0)[0]
    self.assertEqual(expected, s['_id'])

  def test_find_closest_org_names_self(self):
    _threshold = HyperParameters.subsidiary_name_match_min_jaro_similarity
    # finding self
    for s1 in subsidiaries:
      augmented = s1['_id']
      known_org_name, similarity = find_closest_org_name(subsidiaries, augmented, _threshold)
      self.assertIsNotNone(known_org_name, f'{augmented} -> NOTHING {similarity}')
      self.assertEqual(s1['_id'], known_org_name['_id'])

  def test_find_closest_org_uppercased(self):
    _threshold = HyperParameters.subsidiary_name_match_min_jaro_similarity

    # finding uppercased
    for s1 in subsidiaries:
      augmented = s1['_id'].upper()
      known_org_name, similarity = find_closest_org_name(subsidiaries, augmented, _threshold)
      self.assertIsNotNone(known_org_name, f'{augmented} -> NOTHING {similarity}')
      self.assertEqual(s1['_id'], known_org_name['_id'])

  def test_find_closest_org_postfix(self):
    _threshold = HyperParameters.subsidiary_name_match_min_jaro_similarity

    # finding uppercased
    for s1 in subsidiaries:
      augmented = s1['_id'] + ' '
      known_org_name, similarity = find_closest_org_name(subsidiaries, augmented, _threshold)
      self.assertIsNotNone(known_org_name, f'{augmented} -> NOTHING {similarity}')
      self.assertEqual(s1['_id'], known_org_name['_id'])

  def test_find_closest_org_postfix_2(self):
    _threshold = HyperParameters.subsidiary_name_match_min_jaro_similarity

    # finding uppercased
    for s1 in subsidiaries:
      augmented = s1['_id'] + ' '
      known_org_name, similarity = find_closest_org_name(subsidiaries, augmented, _threshold)
      self.assertIsNotNone(known_org_name, f'{augmented} -> NOTHING {similarity}')
      self.assertEqual(s1['_id'], known_org_name['_id'])
      print(known_org_name)

  def test_find_closest_org_names_cut_begin(self):
    _threshold = 0.8

    for s1 in subsidiaries:
      augmented = s1['_id'][1:]
      known_org_name, similarity = find_closest_org_name(subsidiaries, augmented, _threshold)
      self.assertIsNotNone(known_org_name, f'{augmented} -> NOTHING {similarity}')
      self.assertEqual(s1['_id'], known_org_name['_id'])

  def test_ContractAgent_as_lsit(self):
    ca = ContractAgent()
    self.assertEqual(0, len(ca.as_list()))


if __name__ == '__main__':
  unittest.main()
