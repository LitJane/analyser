#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from contract_agents import find_org_names, compare_masked_strings, find_closest_org_name
from contract_parser import ContractDocument3
from gpn.gpn import subsidiaries
from hyperparams import HyperParameters


class ContractAgentsTestCase(unittest.TestCase):

  def ___test_find_agents_a(self):
    # TODO:
    text = """« Квант Доверия » ( Акционерное общество ) , именуемый в дальнейшем « Какашка » , в лице Вице – Президента - управляющего Филиалом « Квант Доверия » ( Акционерное общество ) в г. Полевском Анонимизированного Анонима Анонимыча , действующего на основании Доверенности с одной стороны , и Общество с ограниченной ответственностью « Газпромнефть-Борьба со Злом » , именуемое в дальнейшем « Принципал » , в лице Генерального директора Иванова Ивана Васильевича , действующего на основании Устава , с другой стороны , именуемые в дальнейшем « Стороны » , заключили настоящее Дополнительное соглашение №3 ( далее по тексту – « Дополнительное соглашение » ) к Договору о выдаче банковских гарантий ( далее по тексту – « Договор » ) о нижеследующем :"""
    # TODO: this sentence is not parceable

    text = """
    
     ДОГОВОР № САХ-16/00000/00104/Р.на оказание охранных услуг г. Санкт- Петербург     «27» декабря 2016 год.Общество с ограниченной ответственностью «Газпромнефть-Сахалин», именуемое в дальнейшем «Заказчик», в лице Генерального директора Коробкова Александра Николаевича, действующего на основании Устава, с одной стороны, и Общество с ограниченной ответственностью «Частная охранная организация «СТАФ» (ООО «ЧОО «СТАФ») (Лицензия, серия ЧО № 035162, регистрационный № 629 от 30-11-2015 год, на осуществление частной охранной деятельности, выдана ГУ МВД России по г. Санкт-Петербургу и Ленинградской области, предоставлена на срок до 11-02-2022 года), именуемое в дальнейшем «Исполнитель», в лице Генерального директора Гончарова Геннадия Дмитриевича, действующего на основании Устава, с другой стороны при отдельном упоминании именуемая – Сторона, при совместном упоминании именуемые – Стороны, заключили настоящий договор (далее по тексту – Договор) о нижеследующем1. 
    """

    cd = ContractDocument3(text)
    cd.parse()

    cd.agents_tags = find_org_names(cd)

    _dict = {}
    for tag in cd.agents_tags:
      print(tag)
      _dict[tag.kind] = tag.value

    self.assertIn('org.1.name', _dict)
    self.assertIn('org.2.name', _dict)

    self.assertIn('org.1.alias', _dict)
    self.assertIn('org.2.alias', _dict)

    self.assertIn('org.1.type', _dict)
    self.assertIn('org.2.type', _dict)

    self.assertEqual('Газпромнефть-Борьба со Злом', _dict['org.2.name'])
    self.assertEqual('Квант Доверия', _dict['org.1.name'])
    self.assertEqual('Акционерное Общество', _dict['org.1.type'])
    self.assertEqual('фонд поддержки социальных инициатив', _dict['org.2.type'])

  def test_compare_masked_strings(self):
    similarity = compare_masked_strings('Газпромнефть -  МАбильная карта', 'Газпромнефть-Мобильная карта', [])
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
    known_org_name, _ = find_closest_org_name([mc], 'Газпромнефть - МАбильная карта',
                                              HyperParameters.subsidiary_name_match_min_jaro_similarity)
    self.assertIsNotNone(known_org_name)

    known_org_name, _ = find_closest_org_name(subsidiaries, 'Газпромнефть - МАбильная карта',
                                              HyperParameters.subsidiary_name_match_min_jaro_similarity)
    self.assertIsNotNone(known_org_name)
    self.assertEqual(mc['_id'], known_org_name['_id'])



  def test_compare_masked_strings_1(self):
    s = compare_masked_strings('Многофункциональный комплекс «Лахта центр»', 'Многофункциональный комплекс «Лахта центр»', [])
    print(s)
    for s1 in subsidiaries:
      for name in s1['aliases']:
        s = compare_masked_strings(name, name, [])
        self.assertEqual(s,1)

  def test_find_closest_org_name_solo(self):
    # s = compare_masked_strings('Многофункциональный комплекс «Лахта центр»',
    #                            'Многофункциональный комплекс «Лахта центр»', [])

    s = find_closest_org_name(subsidiaries,  'Многофункциональный комплекс «Лахта центр»', 0)
    print(s)
    # print(s)
    # for s1 in subsidiaries:
    #   for name in s1['aliases']:


  def test_find_closest_org_names_self(self):
    _threshold = HyperParameters.subsidiary_name_match_min_jaro_similarity
    #finding self
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
      self.assertIsNotNone(known_org_name,  f'{augmented} -> NOTHING {similarity}')
      self.assertEqual(s1['_id'], known_org_name['_id'])


  def test_find_closest_org_postfix(self):
    _threshold = HyperParameters.subsidiary_name_match_min_jaro_similarity

    # finding uppercased
    for s1 in subsidiaries:
      augmented = s1['_id'] + ' x'
      known_org_name, similarity = find_closest_org_name(subsidiaries, augmented, _threshold)
      self.assertIsNotNone(known_org_name,  f'{augmented} -> NOTHING {similarity}')
      self.assertEqual(s1['_id'], known_org_name['_id'])

  def test_find_closest_org_postfix_2(self):
    _threshold = HyperParameters.subsidiary_name_match_min_jaro_similarity

    # finding uppercased
    for s1 in subsidiaries:
      augmented = s1['_id'] + ' 2'
      known_org_name, similarity = find_closest_org_name(subsidiaries, augmented, _threshold)
      self.assertIsNotNone(known_org_name,  f'{augmented} -> NOTHING {similarity}')
      self.assertEqual(s1['_id'], known_org_name['_id'])

  def test_find_closest_org_prefix (self):
    _threshold = HyperParameters.subsidiary_name_match_min_jaro_similarity

    # finding uppercased
    for s1 in subsidiaries:
      augmented = 'c' + s1['_id']
      known_org_name, similarity = find_closest_org_name(subsidiaries, augmented, _threshold)
      self.assertIsNotNone(known_org_name,  f'{augmented} -> NOTHING {similarity}')
      self.assertEqual(s1['_id'], known_org_name['_id'])


  def test_find_closest_org_names_cut_begin(self):
    _threshold = 0.8

    for s1 in subsidiaries:
      augmented = s1['_id'][1:]
      known_org_name, similarity = find_closest_org_name(subsidiaries, augmented, _threshold)
      self.assertIsNotNone(known_org_name, f'{augmented} -> NOTHING {similarity}')
      self.assertEqual(s1['_id'], known_org_name['_id'])




if __name__ == '__main__':
  unittest.main()
