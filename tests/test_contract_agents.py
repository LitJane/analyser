#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from contract_agents import find_org_names
from contract_parser import ContractDocument3


class ContractAgentsTestCase(unittest.TestCase):

  def ___test_find_agents_a(self):
    # TODO:
    text = """« Квант Доверия » ( Акционерное общество ) , именуемый в дальнейшем « Какашка » , в лице Вице – Президента - управляющего Филиалом « Квант Доверия » ( Акционерное общество ) в г. Полевском Анонимизированного Анонима Анонимыча , действующего на основании Доверенности с одной стороны , и Общество с ограниченной ответственностью « Газпромнефть-Борьба со Злом » , именуемое в дальнейшем « Принципал » , в лице Генерального директора Иванова Ивана Васильевича , действующего на основании Устава , с другой стороны , именуемые в дальнейшем « Стороны » , заключили настоящее Дополнительное соглашение №3 ( далее по тексту – « Дополнительное соглашение » ) к Договору о выдаче банковских гарантий ( далее по тексту – « Договор » ) о нижеследующем :"""
    # TODO: this sentence is not parceable

    text="""
    
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

  def test_find_agents(self):
    doc_text = """Акционерное общество «Газпром - Вибраниум и Криптонит» (АО «ГВК»), именуемое в \
    дальнейшем «Благотворитель», в лице заместителя генерального директора по персоналу и \
    организационному развитию Неизвестного И.И., действующего на основании на основании Доверенности № Д-17 от 29.01.2018г, \
    с одной стороны, и Фонд поддержки социальных инициатив «Интерстеларные пущи», именуемый в дальнейшем «Благополучатель», \
    в лице Генерального директора ____________________действующего на основании Устава, с другой стороны, \
    именуемые совместно «Стороны», а по отдельности «Сторона», заключили настоящий Договор о нижеследующем:
    """

    cd = ContractDocument3(doc_text)
    cd.parse()

    # agent_infos = find_org_names_spans(cd.tokens_map_norm)
    cd.agents_tags = find_org_names(cd)

    _dict = {}
    for tag in cd.agents_tags:
      print(tag)
      _dict[tag.kind] = tag.value

    self.assertEqual('Акционерное Общество', _dict['org.1.type'])
    self.assertEqual('фонд поддержки социальных инициатив', _dict['org.2.type'])
    self.assertEqual('Интерстеларные пущи', _dict['org.2.name'])
    self.assertEqual('Газпром - Вибраниум и Криптонит', _dict['org.1.name'])

    # self.assertEqual('фонд поддержки социальных инициатив ', cd.agents_tags[6]['value'])


if __name__ == '__main__':
  unittest.main()
