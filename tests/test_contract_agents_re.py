#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import re
import unittest
from typing import List

from analyser.contract_agents import r_type_and_name, complete_re_str, find_org_names, complete_re_ignore_case, \
  complete_re_str_org, complete_re, r_alias, r_alter, r_quoted_name_alias, r_being_a_citizen, r_being_a_human_citizen
from analyser.contract_parser import ContractDocument
from analyser.legal_docs import LegalDocument
from analyser.ml_tools import SemanticTag
from analyser.protocol_parser import ProtocolDocument, find_protocol_org
from analyser.text_normalize import _r_name_ru, r_human_abbr_name, r_human_full_name, _r_name_lat, replacements_regex, \
  r_alias_prefix, r_types, sub_ip_quoter, sub_alias_quote, r_human_name, ru_cap, r_quoted_name, r_group

_suffix = " слово" * 1000


def normalize_contract(_t: str) -> str:
  t = _t
  for (reg, to) in replacements_regex:
    t = reg.sub(to, t)

  return t


def n(x):
  return normalize_contract(x)


class TestContractAgentsSearch(unittest.TestCase):

  def test_ru_cap(self):
    x = ru_cap(n('Государственной автономной учрежденией'))
    self.assertEqual(r'([Гг]осударственн[а-я]{0,3})\s+([Аа]втономн[а-я]{0,3})\s+([Уу]чреждени[а-я]{0,3})', x)

    x = ru_cap('автономной учрежденией')
    self.assertEqual(r'([Аа]втономн[а-я]{0,3})\s+([Уу]чреждени[а-я]{0,3})', x)

  def test_r_name(self):

    r = re.compile(_r_name_ru, re.MULTILINE)
    x = r.search('УУУ')
    print(x)
    x = r.search('ННН')
    print(x)

    r = re.compile(_r_name_lat, re.MULTILINE)
    x = r.search('YYy')
    print(x)

  def test_r_type_and_name(self):

    r = re.compile(r_type_and_name, re.MULTILINE)

    x = r.search(n('Общество с ограниченной ответственностью « Газпромнефть-Региональные продажи » и вообще'))
    self.assertEqual('Общество с ограниченной ответственностью', x['type'])
    self.assertEqual('Газпромнефть-Региональные продажи', x['name'])

    x = r.search(n('Общество с ограниченной ответственностью и прочим «Меццояха» и вообще'))
    self.assertEqual(x['type'], 'Общество с ограниченной ответственностью')
    self.assertEqual(x['name'], 'Меццояха')

    x = r.search('с ООО «УУУ»')
    self.assertEqual('ООО', x['type'])
    self.assertEqual('УУУ', x['name'])

    x = r.search(n('с большой Акционерной обществой  « УУУ » и вообще'))
    self.assertEqual('Акционерной обществой', x['type'])
    self.assertEqual('УУУ', x['name'])

    x = r.search(n('с большой Государственной автономной учрежденией  « УУУ »'))
    self.assertEqual('УУУ', x['name'])
    self.assertEqual('Государственной автономной учрежденией', x['type'])

    x = r.search(n('Общество с ограниченной ответственностью и прочим « Газпромнефть-Региональные продажи » и вообще'))
    self.assertEqual(x['type'], 'Общество с ограниченной ответственностью')
    self.assertEqual(x['name'], 'Газпромнефть-Региональные продажи')

    t = """
       ООО «Газпромнефть-Региональные продажи» в дальнейшем «Благотворитель», с другой стороны
       """
    x = r.search(t)
    for c in range(6):
      print(c, x[c])
    self.assertEqual('ООО', x['type'])
    self.assertEqual('Газпромнефть-Региональные продажи', x['name'])

  def test_r_type_and_name_2(self):

    r = re.compile(r_type_and_name, re.MULTILINE)

    t = """
       ООО «Газпромнефть-Региональные продажи», в дальнейшем «Благотворитель», с другой стороны
       """
    x = r.search(t)
    for c in range(6):
      print(c, x[c])
    self.assertEqual('ООО', x['type'])
    self.assertEqual('Газпромнефть-Региональные продажи', x['name'])

  def test_org_re(self):
    r = re.compile(complete_re_str, re.MULTILINE)

    x = r.search(
      'некое Общество с ограниченной ответственностью «Меццояхве » ( или "Иначе"), именуемое в дальнейшем "Нечто"')
    for c in range(13):
      print(c, x[c])

    self.assertEqual('Общество с ограниченной ответственностью', x['type'])
    self.assertEqual('Меццояхве', x['name'])
    self.assertEqual('( или "Иначе")', x['alt_name'])
    self.assertEqual('Нечто', x['alias'])

  def test_org_re_2(self):
    r = re.compile(complete_re_str, re.MULTILINE)

    x = r.search(
      'некое Общество с ограниченной ответственностью «НЕ БЕЗ НАЗВАНИЯ» ( или "Иначе"), как бе именуемое в дальнейшем "Как-то так"')

    self.assertEqual('Общество с ограниченной ответственностью', x['type'])
    self.assertEqual('НЕ БЕЗ НАЗВАНИЯ', x['name'])
    self.assertEqual('( или "Иначе")', x['alt_name'])
    self.assertEqual('Как-то так', x['alias'])

  def _validate_org(self, tags, org_n, expectation):

    def tag_val(name):
      tag = SemanticTag.find_by_kind(tags, name)
      if tag is not None:
        return tag.value

    self.assertEqual(expectation[1], tag_val(f'org-{org_n}-name'))
    self.assertEqual(expectation[0], tag_val(f'org-{org_n}-type'))
    self.assertEqual(expectation[2], tag_val(f'org-{org_n}-alias'))

  def test_org_dict_0_1(self):

    t0 = """Общество с ограниченной ответственностью «Газпромнефть-сахалин», в лице Генерального директора, Имя Имя Имя, действующего на основании Устава, именуемое в дальнейшем «Заказчик», и ЧАСТНОЕ""" \
         + _suffix
    tags: [SemanticTag] = find_org_names(LegalDocument(t0).parse())

    self._validate_org(tags, 1, ('Общество с ограниченной ответственностью', 'Газпромнефть-Сахалин', 'Заказчик'))

  @unittest.skip("Лицензия is tooo long, need to rewrite RE")
  def test_org_dict_0_2(self):

    t0 = """, и 
    Общество с ограниченной ответственностью «Частная охранная \
    организация «СТАР» (ООО «ЧОО «СТАР») (Лицензия, серия ЧО № _________, регист\
    рационный № ___ от ___________., на осуществление частной охранной деятельности, \
    выдана ГУ МВД России по г. Санкт-Петербургу и Ленинградской области, предоставлена\
     на срок до _________ года), именуемое в дальнейшем «Исполнитель», в лице _______________ Гончаров\
     а Геннадия Федоровича, действующего на основании Устава, с другой стороны """

    tags: [SemanticTag] = find_org_names(LegalDocument(t0).parse())

    self._validate_org(tags, 1, (
      'Общество с ограниченной ответственностью', 'Частная охранная организация «СТАР»', 'Исполнитель'))

  def test_org_dict_0(self):

    t1 = 'ЧАСТНОЕ УЧРЕЖДЕНИЕ ДОПОЛНИТЕЛЬНОГО ПРОФЕССИОНАЛЬНОГО ОБРАЗОВАНИЯ " БИЗНЕС ШКОЛА - СЕМИНАРЫ", ' \
         'действующее на ' \
         'основании лицензии №_____от _______., именуемое в дальнейшем «Исполнитель», ' \
         'в лице ___________ Имя Имя Имя,  ' \
         'действующего на основании Устава, и Имя Имя Имя, именуемая в дальнейшем «Слушатель», в дальнейшем совместно ' \
         'именуемые «Стороны», а по отдельности – «Сторона», заключили настоящий Договор об оказании образовательных ' \
         'услуг (далее – «Договор»)  о нижеследующем:'

    t1 = n(t1)

    r = complete_re_ignore_case
    x = r.search(t1)
    print('===r_alias_prefix=', x['r_alias_prefix'])
    print('===r_quoted_name=', x['r_quoted_name'])
    print('===r_quoted_name_alias=', x['r_quoted_name_alias'])

    tags: List[SemanticTag] = find_org_names(LegalDocument(t1).parse())

    self._validate_org(tags, 1, (
      'ЧАСТНОЕ УЧРЕЖДЕНИЕ ДОПОЛНИТЕЛЬНОГО ПРОФЕССИОНАЛЬНОГО ОБРАЗОВАНИЯ', 'БИЗНЕС ШКОЛА-СЕМИНАРЫ', 'Исполнитель'))

    t = 'Общество с ограниченной ответственностью «Газпромнефть-сахалин», в лице Генерального директора, ' \
        'Имя Имя Имя, действующего на основании Устава, именуемое в дальнейшем «Заказчик», ' \
        'и ЧАСТНОЕ УЧРЕЖДЕНИЕ ДОПОЛНИТЕЛЬНОГО ПРОФЕССИОНАЛЬНОГО ОБРАЗОВАНИЯ " БИЗНЕС ШКОЛА - СЕМИНАРЫ", действующее ' \
        'на основании лицензии №_____от _______., именуемое в дальнейшем «Исполнитель», в лице ___________ Имя Имя Имя, ' \
        'действующего на основании Устава, и Имя Имя Имя, именуемая в дальнейшем «Слушатель», в дальнейшем совместно ' \
        'именуемые «Стороны», а по отдельности – «Сторона», заключили ' \
        'настоящий Договор об оказании образовательных услуг (далее – «Договор») о нижеследующем:'

    tags: List[SemanticTag] = find_org_names(LegalDocument(t).parse())

    self._validate_org(tags, 2, ('Общество с ограниченной ответственностью', 'Газпромнефть-Сахалин', 'Заказчик'))
    self._validate_org(tags, 1, (
      'ЧАСТНОЕ УЧРЕЖДЕНИЕ ДОПОЛНИТЕЛЬНОГО ПРОФЕССИОНАЛЬНОГО ОБРАЗОВАНИЯ', 'БИЗНЕС ШКОЛА-СЕМИНАРЫ', 'Исполнитель'))

  def test_org_dict_1(self):

    t = n("ООО «Газпромнефть-Региональные продажи», в лице начальника управления по связям с общественностью "
          "Иванова Семена Евгеньевича, действующего на основании Доверенности в дальнейшем «Благотворитель», "
          "с другой стороны заключили настоящий Договор о нижеследующем: \n")

    tags: List[SemanticTag] = find_org_names(LegalDocument(t).parse())

    self._validate_org(tags, 1, (
      'Общество с ограниченной ответственностью', 'Газпромнефть-Региональные продажи', 'Благотворитель'))

  def test_org_dict_2(self):

    t = n("""
    ООО «Газпромнефть-Региональные продажи» в дальнейшем «БлаготворЮтель», с другой стороны
    """)

    tags: List[SemanticTag] = find_org_names(LegalDocument(t).parse())
    self._validate_org(tags, 1, (
      'Общество с ограниченной ответственностью', 'Газпромнефть-Региональные продажи', 'БлаготворЮтель'))

  def test_org_dict_3(self):

    t = n("""
    Муниципальное бюджетное учреждение города Москвы «Радуга» именуемый в дальнейшем
    «Благополучатель», в лице директора Соляной Марины Александровны, действующая на основании
    Устава, с одной стороны, и ООО «Газпромнефть-Региональные продажи», аааааааа аааа в дальнейшем «Благотворитель», с другой стороны заключили настоящий Договор о
    нижеследующем:
    """)

    tags: [SemanticTag] = find_org_names(LegalDocument(t).parse(), decay_confidence=False)
    self._validate_org(tags, 2, ('Муниципальное бюджетное учреждение', 'Радуга', 'Благополучатель'))
    self._validate_org(tags, 1, (
      'Общество с ограниченной ответственностью', 'Газпромнефть-Региональные продажи', 'Благотворитель'))

  def test_org_dict_3_1(self):

    t = n("""
    Федеральное государственное бюджетное образовательное учреждение высшего образования «Государственный университет» (ФГБОУ ВО «ГУ»), именуемое в дальнейшем «Исполнитель», в лице ____________________ Сергеева
    """)

    tags: List[SemanticTag] = find_org_names(LegalDocument(t).parse())
    self._validate_org(tags, 1, ('Федеральное государственное бюджетное образовательное учреждение высшего образования',
                                 'Государственный университет', 'Исполнитель'))

  def test_find_ip(self):

    t = n("""
    Сибирь , и Индивидуальный предприниматель « Лужин В. В. » , именуемый в дальнейшем « Исполнитель » , \
    с другой стороны , именуемые в дальнейшем совместно « Стороны » , а по отдельности - « Сторона » , заключили настоящий договор о нижеследующем : 
    """)

    tags: List[SemanticTag] = find_org_names(LegalDocument(t).parse())
    self._validate_org(tags, 1, ('Индивидуальный предприниматель', 'Лужин В. В.', 'Исполнитель'))

  def test_find_ip2(self):

    t = n("""
    Сибирь , и Индивидуальный предприниматель Лужин В. В., именуемый в дальнейшем « Исполнитель » , \
    с другой стороны , именуемые в дальнейшем совместно « Стороны » , а по отдельности - « Сторона » , заключили настоящий договор о нижеследующем : 
    """)

    tags: [SemanticTag] = find_org_names(LegalDocument(t).parse())
    self._validate_org(tags, 1, ('Индивидуальный предприниматель', 'Лужин В. В.', 'Исполнитель'))

  def test_find_ip3(self):

    t = n("""
    Сибирь , и ИП Лужин В. В., именуемый в дальнейшем « Исполнитель » , \
    с другой стороны , именуемые в дальнейшем совместно « Стороны » , а по отдельности - « Сторона » , заключили настоящий договор о нижеследующем : 
    """)

    tags: [SemanticTag] = find_org_names(LegalDocument(t).parse())
    self._validate_org(tags, 1, ('Индивидуальный предприниматель', 'Лужин В. В.', 'Исполнитель'))

  def test_org_dict_4_1(self):

    t = n(
      """Автономная некоммерческая организация дополнительного профессионального образования «ООО»,  \
      именуемое далее Исполнитель, в лице Директора Уткиной Е.В., действующей на основании Устава, с одной стороны,""")

    tags: List[SemanticTag] = find_org_names(LegalDocument(t).parse())
    self._validate_org(tags, 1, (
      'Автономная некоммерческая организация', 'ООО', 'Исполнитель'))

  def test_find_agents_chu(self):
    t = 'ДОГОВОР НА ОКАЗАНИЕ ОБРАЗОВАТЕЛЬНЫХ УСЛУГ № 1449\nГород Москва\t03.10.2016\nОбщество с ограниченной ' \
        'ответственностью «Радость-Радость», в лице Генерального директора, Александра Александра Александра, действу' \
        'ющего на основании Устава, именуемое ' \
        'в дальнейшем «Заказчик», и ЧАСТНОЕ УЧРЕЖДЕНИЕ ДОПОЛНИТЕЛЬНОГО ПРОФЕССИОНАЛЬНОГО ОБРАЗОВАНИЯ " БИЗНЕС ШКОЛА - СЕМИНАРЫ", действующее ' \
        'на основании лицензии №_____от _______., именуемое в ' \
        'дальнейшем «Исполнитель», в лице ___________ Александры Александры Александры, ' \
        'действующего на основании Устава, и '

    tags: List[SemanticTag] = find_org_names(ContractDocument(t).parse())
    self._validate_org(tags, 2, ('Общество с ограниченной ответственностью', 'Радость-Радость', 'Заказчик'))
    self._validate_org(tags, 1, (
      'ЧАСТНОЕ УЧРЕЖДЕНИЕ ДОПОЛНИТЕЛЬНОГО ПРОФЕССИОНАЛЬНОГО ОБРАЗОВАНИЯ', 'БИЗНЕС ШКОЛА-СЕМИНАРЫ', 'Исполнитель'))

  def test_find_agents_1(self):
    doc_text = """Акционерное общество «Газпромнефть - мобильная карта» (АО «ГВК»), именуемое в \
    дальнейшем «Благотворитель», в лице заместителя генерального директора по персоналу и \
    организационному развитию Неизвестного И.И., действующего на основании на основании Доверенности № Д-17 от 29.01.2018г, \
    с одной стороны, и Фонд поддержки социальных инициатив «Интерстеларные пущи», именуемый в дальнейшем «Благополучатель», \
    в лице Генерального директора ____________________действующего на основании Устава, с другой стороны, \
    именуемые совместно «Стороны», а по отдельности «Сторона», заключили настоящий Договор о нижеследующем:
    """

    tags: List[SemanticTag] = find_org_names(LegalDocument(doc_text).parse())
    self._validate_org(tags, 1, ('Акционерное общество', 'Газпромнефть-Мобильная карта', 'Благотворитель'))
    self._validate_org(tags, 2, ('Фонд поддержки социальных инициатив', 'Интерстеларные пущи', 'Благополучатель'))

  def test_find_protocol_org(self):
    doc_text = """ПРОТОКОЛ ВНЕОЧЕРЕДНОГО ОБЩЕГО СОБРАНИЯ УЧАСТНИКОВ ОБЩЕСТВА С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ «Газпромнефть Шиппинг» (далее – «Общество»)
Место проведения внеочередного общего собрания участников:.
Дата и время проведения общего собрания участников: «10» февраля 2017 года, 12 часов 00 минут.
Форма проведения очередного общего собрания участников: в форме собрания (совместного присутствия участников для обсуждения вопросов повестки дня и принятия решения по вопросам, поставленным на голосование). 

Присутствовали участники:
1. ПАО «Газпром нефть», владеющее долей ХХХ % от уставного капитала, в лице Заместителя генерального директора по правовым и корпоративным вопросам ХХХХ, действующего на основании Доверенности № от г. ;
2. ООО «ГПН-ГПН», владеющее долей ХХХ % от уставного капитала, в лице Генерального директора ХХХХ, действующего на основании Устава.
Общее собрание участников правомочно принимать решения по всем вопросам повестки дня.
Председателем собрания единогласно избрана ХХХ.
Секретарём собрания единогласно избран ХХХ. 
Повестка дня:
1. Принятие решения о совершении сделки с ООО «Шторм»

Принятое решение по первому"""

    doc = ProtocolDocument(LegalDocument(doc_text))
    doc.parse()
    print(doc.text)
    tags: [SemanticTag] = find_protocol_org(doc)
    self._validate_org(tags, 1, ('Общество с ограниченной ответственностью', 'Газпромнефть Шиппинг', None))

  def test_find_agents_2(self):
    doc_text = """Акционерное общество «Газпромнефть - мобильная карта» (АО «ГВК»), именуемое в \
    дальнейшем «Благотворитель», в лице заместителя генерального директора по персоналу и \
    организационному развитию Неизвестного И.И., действующего на основании на основании Доверенности № Д-17 от 29.01.2018г, \
    с одной стороны, и Фонд поддержки социальных инициатив «Лингвистическая школа «Слово», именуемый в дальнейшем «Благополучатель», \
    в лице Генерального директора ____________________действующего на основании Устава, с другой стороны, \
    именуемые совместно «Стороны», а по отдельности «Сторона», заключили настоящий Договор о нижеследующем:
    """

    tags: List[SemanticTag] = find_org_names(LegalDocument(doc_text).parse())
    self._validate_org(tags, 1, ('Акционерное общество', 'Газпромнефть-Мобильная карта', 'Благотворитель'))
    self._validate_org(tags, 2,
                       ('Фонд поддержки социальных инициатив', 'Лингвистическая школа «Слово»', 'Благополучатель'))

  def test_find_agents_person(self):
    doc_text = """Общество с ограниченной ответственностью «Кишки Бога» (ООО «Кишки Бога»), именуемое в дальнейшем «Заказчик», \
    в лице генерального директора Шприца Александра Устыныча, действующего на основании Устава, с одной \
    стороны, и Базедов Болезнь Бледнович, являющийся гражданином Российской Федерации, действующий \
    от собственного имени, именуемый в дальнейшем «Исполнитель», с другой стороны, совместно \
    именуемые «Стороны», и каждая в отдельности «Сторона», заключили настоящий """

    tags: List[SemanticTag] = find_org_names(LegalDocument(doc_text).parse())
    self._validate_org(tags, 2, ('Общество с ограниченной ответственностью', 'Кишки Бога', 'Заказчик'))
    self._validate_org(tags, 1, (None, 'Базедов Болезнь Бледнович', 'Исполнитель'))

  def test_find_agent_0(self):
    txt = '''
    , и
    Общество с ограниченной ответственностью «Научно-производственная компания «НефтеБурГаз», в лице Генерального директора Рожкова Александра Владимировича, действующего на основании Устава, именуемое в дальнейшем «Подрядчик»,
    '''

    r = re.compile(r_type_and_name, re.MULTILINE)

    x = r.search(n(txt))
    self.assertEqual('Общество с ограниченной ответственностью', x['type'])
    self.assertEqual('Научно-производственная компания «НефтеБурГаз', x['name'])

    tags: List[SemanticTag] = find_org_names(LegalDocument(txt).parse())
    self._validate_org(tags, 1, (
      'Общество с ограниченной ответственностью', 'Научно-производственная компания «НефтеБурГаз»', 'Подрядчик'))

  def test_find_agent_no_comma(self):
    txt_full = 'Акционерное Общество «Газпромнефть – Терминал» именуемое в дальнейшем «Продавец», в лице генерального ' \
               'директора, действующего на основании Устава, с одной стороны, и ООО «Ромашка», именуемое в ' \
               'дальнейшем «Покупатель», в лице Петрова П.П., действующего на основании Устава, с другой стороны, совместно ' \
               'именуемые «Стороны», а по отдельности - «Сторона», заключили настоящий договор (далее по тексту – ' \
               'Договор) о нижеследующем:'
    doc = LegalDocument(txt_full).parse()
    print(doc.text)

    txt = txt_full[150:]

    r = re.compile(r_quoted_name, re.MULTILINE)
    normalized_txt = n(txt)
    x = r.search(normalized_txt)
    self.assertEqual('Ромашка', x['name'])

    r = re.compile(r_type_and_name, re.MULTILINE)
    x = r.search(normalized_txt)
    self.assertEqual('ООО', x['type'])
    self.assertEqual('Ромашка', x['name'])

    r = re.compile(complete_re_str_org, re.MULTILINE)
    x = r.search(normalized_txt)
    self.assertEqual('ООО', x['type'])
    self.assertEqual('Ромашка', x['name'])

    r = re.compile(complete_re_str, re.MULTILINE)
    x = r.search(normalized_txt)
    self.assertEqual('ООО', x['type'])
    self.assertEqual('Ромашка', x['name'])
    self.assertEqual('Покупатель', x['alias'])
    print('r_alias_prefix=', x['r_alias_prefix'])
    print('_alias_ext=', x['_alias_ext'])

    r = complete_re
    x = r.search(normalized_txt)
    self.assertEqual('ООО', x['type'])
    self.assertEqual('Ромашка', x['name'])

    tags: List[SemanticTag] = find_org_names(doc, decay_confidence=False)
    self._validate_org(tags, 1, ('Акционерное общество', 'Газпромнефть-Терминал', 'Продавец'))
    self._validate_org(tags, 2, ('Общество с ограниченной ответственностью', 'Ромашка', 'Покупатель'))

  def test_find_agent_ao(self):
    txt = '''Акционерное Общество «Газпромнефть – Терминал» именуемое в дальнейшем «Продавец», \
      в лице генерального директора, действующего на основании Устава, с одной стороны, и ООО «Ромашка»'''

    r = re.compile(r_quoted_name, re.MULTILINE)
    x = r.search(n(txt))
    self.assertEqual('Газпромнефть – Терминал', x['name'])

    r = re.compile(r_type_and_name, re.MULTILINE)
    x = r.search(n(txt))
    self.assertEqual('Акционерное Общество', x['type'])
    self.assertEqual('Газпромнефть – Терминал', x['name'])

    r = re.compile(complete_re_str_org, re.MULTILINE)
    x = r.search(n(txt))
    self.assertEqual('Акционерное Общество', x['type'])
    self.assertEqual('Газпромнефть – Терминал', x['name'])

    r = re.compile(complete_re_str, re.MULTILINE)
    x = r.search(n(txt))
    self.assertEqual('Акционерное Общество', x['type'])
    self.assertEqual('Газпромнефть – Терминал', x['name'])

    print('r_alias_prefix=', x['r_alias_prefix'])
    print('_alias_ext=', x['_alias_ext'])

    r = complete_re
    x = r.search(n(txt))
    self.assertEqual('Акционерное Общество', x['type'])
    self.assertEqual('Газпромнефть – Терминал', x['name'])

    tags: List[SemanticTag] = find_org_names(LegalDocument(txt).parse(), decay_confidence=False)
    self._validate_org(tags, 1, ('Акционерное общество', 'Газпромнефть-Терминал', 'Продавец'))

  def test_find_agent_fond(self):
    txt = '''Одобрить предоставление безвозмездной финансовой помощи Фонду «Олимп» в размере 1500000 (один миллион пятьсот тысяч) рублей \
     для создания и организации работы интернет-платформы «Олимп» по поддержке стартапов в сфере взаимопомощи.
    Время подведения итогов голосования – 18 часов 00 минут.'''

    r = re.compile(r_quoted_name, re.MULTILINE)
    x = r.search(n(txt))
    self.assertEqual('Олимп', x['name'])

    r = re.compile(r_type_and_name, re.MULTILINE)
    x = r.search(n(txt))
    self.assertEqual('Фонду', x['type'])
    self.assertEqual('Олимп', x['name'])

    r = re.compile(complete_re_str_org, re.MULTILINE)
    x = r.search(n(txt))
    self.assertEqual('Фонду', x['type'])
    self.assertEqual('Олимп', x['name'])

    r = re.compile(complete_re_str, re.MULTILINE)
    x = r.search(n(txt))
    self.assertEqual('Фонду', x['type'])
    self.assertEqual('Олимп', x['name'])

    r = complete_re
    x = r.search(n(txt))
    self.assertEqual('Фонду', x['type'])
    self.assertEqual('Олимп', x['name'])

    tags: List[SemanticTag] = find_org_names(LegalDocument(txt).parse(), decay_confidence=False)
    self._validate_org(tags, 1, (
      'Фонду', 'Олимп', None))

  def test_find_agent_fond_2(self):
    txt = '''           основании Устава, с одной стороны, и Фонд «Благо», именуемое в дальнейшем «Благополучатель», в лице председателя'''

    r = re.compile(r_quoted_name, re.MULTILINE)
    x = r.search(n(txt))
    self.assertEqual('Благо', x['name'])

    r = re.compile(r_type_and_name, re.MULTILINE)
    x = r.search(n(txt))
    self.assertEqual('Фонд', x['type'])
    self.assertEqual('Благо', x['name'])

    r = re.compile(complete_re_str_org, re.MULTILINE)
    x = r.search(n(txt))
    self.assertEqual('Фонд', x['type'])
    self.assertEqual('Благо', x['name'])

    r = re.compile(complete_re_str, re.MULTILINE)
    x = r.search(n(txt))
    self.assertEqual('Фонд', x['type'])
    self.assertEqual('Благо', x['name'])

    r = complete_re
    x = r.search(n(txt))
    self.assertEqual('Фонд', x['type'])
    self.assertEqual('Благо', x['name'])

    tags: List[SemanticTag] = find_org_names(LegalDocument(txt).parse(), decay_confidence=False)
    self._validate_org(tags, 1, (
      'Фонд', 'Благо', 'Благополучатель'))

  def test_find_agent_2(self):
    txt1 = '''Общество с ограниченной ответственностью «Комплекс Галерная 5», являющееся юридическим лицом, именуемое в дальнейшем «Принципал»'''
    txt2 = '''
        , и
        Общество с ограниченной ответственностью «Научно-производственная компания «НефтеБурГаз», в лице Генерального директора Рожкова Александра Владимировича, действующего на основании Устава, именуемое в дальнейшем «Подрядчик»,
        '''
    txt = txt1 + txt2
    r = re.compile(r_type_and_name, re.MULTILINE)

    x = r.search(n(txt))
    self.assertEqual('Общество с ограниченной ответственностью', x['type'])
    self.assertEqual('Комплекс Галерная 5', x['name'])

    tags: List[SemanticTag] = find_org_names(LegalDocument(txt).parse())
    self._validate_org(tags, 1, ('Общество с ограниченной ответственностью', 'Комплекс Галерная 5', 'Принципал'))
    self._validate_org(tags, 2, (
      'Общество с ограниченной ответственностью', 'Научно-производственная компания «НефтеБурГаз»', 'Подрядчик'))

  def test_find_agent_ONPZ(self):
    txt = '''
      2016 год.
     Акционерное общество “Газпромнефть-Омский НПЗ” (АО “Газпромнефть-ОНПЗ”), именуемое в дальнейшем «Организацией» водопроводно-канализационного хозяйства'''

    txt = n(txt)

    x = re.compile(r_type_and_name, re.MULTILINE).search(n(txt))
    self.assertEqual('Акционерное общество', x['type'])
    self.assertEqual('Газпромнефть-Омский НПЗ', x['name'])

    tags: List[SemanticTag] = find_org_names(LegalDocument(txt).parse())
    self._validate_org(tags, 1, ('Акционерное общество', 'Газпромнефть-ОНПЗ', 'Организацией'))

  def test_find_agent_MPZ(self):
    txt = '''
      2016 год.
     Акционерное общество “Газпромнефть - МНПЗ” (АО “ГПН-МНПЗ”), именуемое в дальнейшем «Организацией» водопроводно-канализационного хозяйства'''

    txt = n(txt)

    x = re.compile(r_type_and_name, re.MULTILINE).search(n(txt))
    self.assertEqual('Акционерное общество', x['type'])
    self.assertEqual('Газпромнефть - МНПЗ', x['name'])

    tags: List[SemanticTag] = find_org_names(LegalDocument(txt).parse())
    self._validate_org(tags, 1, ('Акционерное общество', 'Газпромнефть-МНПЗ', 'Организацией'))

  def test_org_dict_4_2(self):
    t = n(
      """Государственное автономное  учреждение дополнительного профессионального образования Свердловской области «Армавирский учебно-технический центр»,  на основании Лицензии на право осуществления образовательной деятельности в лице директора  Птицына Евгения Георгиевича, действующего на основании Устава, с одной стороны, """)

    tags: List[SemanticTag] = find_org_names(LegalDocument(t).parse())
    self._validate_org(tags, 1, ('Государственное автономное учреждение', 'Армавирский учебно-технический центр', None))

  def test_org_dict(self):

    t = """
    Муниципальное бюджетное учреждение города Москвы «Радуга» именуемый в дальнейшем
    «Благополучатель», в лице директора Соляной Марины Александровны, действующая на основании
    Устава, с одной стороны, и 
    
    ООО «Газпромнефть-Региональные продажи» в лице начальника управления по связям с общественностью Иванова Семена Евгеньевича, действующего на основании Доверенности в дальнейшем «Благотворитель», с другой стороны заключили настоящий Договор о нижеследующем:
    """ + _suffix

    tags: List[SemanticTag] = find_org_names(LegalDocument(t).parse())
    self._validate_org(tags, 2, ('Муниципальное бюджетное учреждение', 'Радуга', 'Благополучатель'))
    self._validate_org(tags, 1, (
      'Общество с ограниченной ответственностью', 'Газпромнефть-Региональные продажи', 'Благотворитель'))

  def test_r_types(self):
    r = re.compile(r_types, re.MULTILINE)

    x = r.search('с большой Общество с ограниченной ответственностью « Газпромнефть-Региональные продажи »')
    self.assertEqual('Общество с ограниченной ответственностью', x['type'])

    x = r.search('Общество с ограниченной ответственностью и прочим « Газпромнефть-Региональные продажи »')
    self.assertEqual('Общество с ограниченной ответственностью', x['type'])

    x = r.search('ООО « Газпромнефть-Региональные продажи »')
    self.assertEqual('ООО', x['type'])

    x = r.search('с большой Государственной автономной учрежденией\n')
    self.assertEqual('Государственной автономной учрежденией', x['type'])

    x = r.search('акционерное Общество с ограниченной ответственностью и прочим « Газпромнефть-Региональные продажи »')
    self.assertEqual('акционерное Общество', x['type'])

    x = r.search('АО  и прочим « Газпромнефть-Региональные продажи »')
    self.assertEqual('АО', x['type'])

    x = r.search('префикс АО  и прочим « Газпромнефть-Региональные продажи »')
    self.assertEqual('АО', x['type'])

  def test_r_alias_prefix(self):
    r = re.compile(r_alias_prefix, re.MULTILINE)
    print(r)

    x = r.search('что-то именуемое в дальнейшем Жертвователь, и ')
    self.assertEqual(x[2], 'именуемое в дальнейшем ')

    x = r.search('именуемое далее Жертвователь или еще как')
    self.assertEqual(x[2], 'именуемое далее ')

    x = r.search('в дальнейшем  Жертвователь или еще как')
    self.assertEqual(x[2], 'в дальнейшем ')

    x = r.search('нечто, именуемое далее Жертвователь или еще как')
    self.assertEqual(x[2], 'именуемое далее ')

    x = r.search('нечто, далее - Жертвователь или еще как')
    self.assertEqual(x[1], ' далее -')

  def test_r_alias(self):
    r = re.compile(r_alias, re.MULTILINE)

    x = r.search('что-то, именуемое в дальнейшем « Жертвователь » или иначе')
    self.assertEqual('Жертвователь', x['alias'])

    x = r.search('что-то именуемое в дальнейшем «Жертвователь-какаха», и ')
    self.assertEqual('Жертвователь-какаха', x['alias'])

  def test_r_alias_1(self):
    t = '''" БИЗНЕС ШКОЛА - СЕМИНАРЫ", действующее на основании "
              лицензии №_____от _______., именуемое в дальнейшем «Исполнитель», в лице ___________ Имя Имя Имя'''

    r = re.compile(r_alter + r_alias, re.MULTILINE)
    x = r.search(t)

    print('r_alias_prefix=', x['r_alias_prefix'])
    self.assertEqual('Исполнитель', x['alias'])

  def test_r_alias_2(self):
    t = """Общество с ограниченной ответственностью «Газпромнефть-Захалин», в лице Генерального директора, "
               "Коробкова Александра Николаевича, действующего на основании Устава, именуемое в дальнейшем «Заказчик», 
               и ЧАСТНОЕ УЧРЕЖДЕНИЕ ДОПОЛНИТЕЛЬНОГО ПРОФЕССИОНАЛЬНОГО ОБРАЗОВАНИЯ " БИЗНЕС ШКОЛА - СЕМИНАРЫ", действующее на основании "
               лицензии №_____от _______., именуемое в дальнейшем «Исполнитель», в лице ___________ Самойловой Александры Николаевны, "
               действующего на основании Устава, и Сергеева Ольга Георгиевна, именуемая в дальнейшем «Слушатель», в дальнейшем совместно "
               именуемые «Стороны», а по отдельности – «Сторона», заключили настоящий Договор об оказании образовательных услуг (далее – «Договор») " 
               о нижеследующем:"""

    r = re.compile(r_alias, re.MULTILINE)

    x = r.search(t)
    self.assertEqual('Заказчик', x['alias'])

  def test_r_quoted_name(self):

    t = n("""
        ООО «Газпромнефть-Региональные продажи» в дальнейшем «Благотворитель», с другой стороны
        """)

    r1 = re.compile(r_quoted_name, re.MULTILINE)
    x1 = r1.search(t)

    self.assertEqual('Газпромнефть-Региональные продажи', x1['name'])

  def test_r_quoted_name_2(self):

    t = n("Фонд поддержки социальных инициатив «Лингвистическая школа «Слово», именуемый")

    r1 = re.compile(r_quoted_name, re.MULTILINE)
    x1 = r1.search(t)

    self.assertEqual('Лингвистическая школа «Слово', x1['name'])

  def test_r_quoted_name_alias(self):

    rgc = re.compile(r_quoted_name_alias, re.MULTILINE)

    x = rgc.search('что-то именуемое в дальнейшем " Абралябмда филорна", и ')
    self.assertEqual('Абралябмда филорна', x['alias'])

    x = rgc.search('что-то именуемое в дальнейшем «Жертвователь», и ')
    self.assertEqual('Жертвователь', x['alias'])

  def test_replace_alias(self):
    r = sub_alias_quote

    t = 'что-то именуемое в дальнейшем Жертвователь-какаха, и нечто, именуемое далее КАКАХА-ХА '

    x = sub_alias_quote[0].search(t)
    for c in range(7):
      print(c, x[c])

    replacer = r[1]
    pattern = r[0]

    xx = pattern.sub(replacer, t)
    self.assertEqual('что-то именуемое в дальнейшем «Жертвователь-какаха», и нечто, именуемое далее «КАКАХА-ХА» ', xx)

    xx = pattern.sub(replacer, 'что-то в дальнейшем Жертвователь-какаха, и нечто, именуемое далее КАКАХА-ХА')
    self.assertEqual('что-то в дальнейшем «Жертвователь-какаха», и нечто, именуемое далее «КАКАХА-ХА»', xx)

    xx = pattern.sub(replacer, 'что-то, далее - Благожертвователь-какаха, и нечто')
    self.assertEqual('что-то, далее - «Благожертвователь-какаха», и нечто', xx)

    xx = pattern.sub(replacer, 'что-то, далее именуемое Благожертвователь-какаха, и нечто')
    self.assertEqual('что-то, далее именуемое «Благожертвователь-какаха», и нечто', xx)

    xx = pattern.sub(replacer, 'далее - Благожертвователь-какаха, и нечто')
    self.assertEqual('далее - «Благожертвователь-какаха», и нечто', xx)

  def test_r_human_full_name(self):
    r = re.compile(r_human_full_name, re.MULTILINE)

    x = r.search('что-то Мироздания Леонидыч Крупица, Который "был" таков')
    self.assertEqual('Мироздания Леонидыч Крупица', x[1])

    x = r.search('что-то Мироздания Крупица, который был')
    self.assertEqual('Мироздания Крупица', x[1])

    x = r.search('Мироздания Крупица , который был')
    self.assertEqual('Мироздания Крупица', x[1])

    x = r.search('что-то Абрам, который был')
    self.assertEqual(None, x)

  def test_r_being_a_citizen3(self):
    t0 = ''' именуемое в дальнейшем «Заказчик», в лице генерального директора Пулькина Фасо Кларобыча, действующего \
    на основании Устава, с одной стороны, и Фамильный Имен Отчестыч, являющийся гражданином Российской Федерации, \
    действующий от собственного имени, именуемый в дальнейшем «Исполнитель», с другой стороны, совместно \
    именуемые «Стороны», и каждая в отдельности «Сторона», заключили настоящий'''

    t1 = ''' именуемое в дальнейшем «Заказчик», в лице генерального директора Пулькина Фасо Кларобыча, действующего \
        на основании Устава, с одной стороны, и Фамильная Именка Отчестычвовнв, являющаяся гражданинкой Некой Страны, \
        действующий от собственного имени, именуемый в дальнейшем «Исполнитель», с другой стороны, совместно \
        именуемые «Стороны», и каждая в отдельности «Сторона», заключили настоящий'''
    r = re.compile(r_being_a_citizen, re.MULTILINE)

    x = r.search(t0)
    self.assertEqual('являющийся гражданином', x['citizen'])

    x = r.search(t1)
    self.assertEqual('являющаяся гражданинкой', x['citizen'])

  def test_find_agents_personz_1(self):
    t0 = """с одной \
    стороны, и Базедов Недуг Бледнович, являющийся гражданином Российской Федерации, действующий \
    от собственного имени, именуемый в дальнейшем «Исполнитель», с другой стороны, совместно \
    именуемые «Стороны», и каждая в отдельности «Сторона», заключили настоящий """

    r = re.compile(complete_re_str, re.MULTILINE)
    x = r.search(t0)
    for t in x.groups():
      print(t)
    self.assertEqual('Базедов Недуг Бледнович', x['human_name'])

  def test_find_agents_personz_2(self):
    t0 = """Общество с ограниченной ответственностью «Кишки Бога» (ООО «Кишки Бога»), именуемое в дальнейшем «Заказчик», \
    в лице генерального директора Шприца Александра Устыныча, действующего на основании Устава, с одной \
    стороны, и Базедов Недуг Бледнович, являющийся гражданином Российской Федерации, действующий \
    от собственного имени, именуемый в дальнейшем «Исполнитель», с другой стороны, совместно \
    именуемые «Стороны», и каждая в отдельности «Сторона», заключили настоящий """

    tags: List[SemanticTag] = find_org_names(LegalDocument(t0).parse())
    self._validate_org(tags, 2, ('Общество с ограниченной ответственностью', 'Кишки Бога', 'Заказчик'))
    self._validate_org(tags, 1, (None, 'Базедов Недуг Бледнович', 'Исполнитель'))

  def test_r_human_citizen(self):
    t0 = ''' в лице генерального директора Пулькина Фасо Кларобыча, действующего \
    на основании Устава, с одной стороны, и Фамильный Имен Отчестыч, являющийся гражданином Российской Федерации, \
    действующий от собственного имени, именуемый в дальнейшем «Исполнитель», с другой стороны, совместно \
    именуемые «Стороны», и каждая в отдельности «Сторона», заключили настоящий'''

    r = re.compile(r_being_a_human_citizen, re.MULTILINE)

    x = r.search(t0)
    self.assertEqual('Фамильный Имен Отчестыч, являющийся гражданином', x['human_citizen'])

  def test_r_human_abbr_name(self):
    r = re.compile(r'\W' + r_human_abbr_name, re.MULTILINE)

    x = r.search('что-то Мироздания С.К., который был')
    self.assertEqual('Мироздания С.К.', x[1])

    x = r.search('что-то Мироздания С. К., который был')
    self.assertEqual('Мироздания С. К.', x[1])

    x = r.search('что-то Мироздания С. , который был')
    self.assertEqual('Мироздания С. ', x[1])

    x = r.search('что-то  С. , который был')
    self.assertEqual(None, x)

    x = r.search('что-то друГое  С. R. , который был')
    self.assertEqual(None, x)

  def test_r_human_name(self):

    r = re.compile(r'\W' + r_group(r_human_abbr_name, 'human_name'), re.MULTILINE)

    x = r.search('что-то Газпромнефть-Омский НПЗ, который был')
    self.assertIsNone(x)

    r = re.compile(r'\W' + r_group(r_human_full_name, 'human_name'), re.MULTILINE)
    x = r.search('что-то Газпромнефть-Омский НПЗ, который был')
    self.assertIsNone(x)

    r = re.compile(r'\W' + r_human_name, re.MULTILINE)
    x = r.search('что-то Мироздания С.К., который был')
    self.assertEqual('Мироздания С.К.', x['human_name'])

    x = r.search('что-то Мироздания С. К., который был')
    self.assertEqual('Мироздания С. К.', x['human_name'])

    x = r.search('что-то Мироздания С. , который был')
    self.assertEqual('Мироздания С. ', x['human_name'])

    x = r.search('что-то  С. , который был')
    self.assertEqual(None, x)

    x = r.search('что-то друГое  С. R. , который был')
    self.assertEqual(None, x)

    x = r.search('что-то Мироздания Леонидыч Крупица, Который "был" таков')
    self.assertEqual('Мироздания Леонидыч Крупица', x['human_name'])

    x = r.search('что-то Мироздания Крупица, который был')
    self.assertEqual('Мироздания Крупица', x['human_name'])

  def test_replace_ip(self):
    r = sub_ip_quoter

    replacer = r[1]
    pattern = r[0]

    xx = pattern.sub(replacer, 'ИП Петров В.К.')
    self.assertEqual('ИП «Петров В.К.»', xx)

    xx = pattern.sub(replacer, 'Индивидуальная предприниматель Петров В.К.')
    self.assertEqual('Индивидуальная предприниматель «Петров В.К.»', xx)

    xx = pattern.sub(replacer, 'некогда ИП Петров В.К.')
    self.assertEqual('некогда ИП «Петров В.К.»', xx)

    xx = pattern.sub(replacer, 'некогда Индивидуальная предприниматель Петров В.К.')
    self.assertEqual('некогда Индивидуальная предприниматель «Петров В.К.»', xx)

    xx = pattern.sub(replacer, 'некогда ИП Петров В.К.')
    self.assertEqual('некогда ИП «Петров В.К.»', xx)

    xx = pattern.sub(replacer, 'некогда Индивидуальная предприниматель Мироздания Леонидыч Крупица')
    self.assertEqual('некогда Индивидуальная предприниматель «Мироздания Леонидыч Крупица»', xx)

    xx = pattern.sub(replacer, 'некогда ПРИНЦИПАЛ Мироздания Леонидыч Крупица')
    self.assertEqual('некогда ПРИНЦИПАЛ Мироздания Леонидыч Крупица', xx)

    t = """директора Регион Сибирь, и Индивидуальный предприниматель Петров В.В., именуемый в дальнейшем «Исполнитель»"""
    xx = pattern.sub(replacer, t)
    self.assertEqual(
      'директора Регион Сибирь, и Индивидуальный предприниматель «Петров В.В.», именуемый в дальнейшем «Исполнитель»',
      xx)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
