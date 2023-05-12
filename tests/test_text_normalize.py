#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from analyser.contract_agents import normalize_legal_entity_type
from analyser.documents import CaseNormalizer, TextMap
from analyser.legal_docs import LegalDocument
from analyser.structures import legal_entity_types
from analyser.text_normalize import normalize_company_name, normalize_text, replacements_regex


class CaseNormalizerTestCase(unittest.TestCase):
  def test_basics(self):
    cn = CaseNormalizer()
    print(cn.normalize_tokens(['стороны', 'Заключили', 'договор', 'уррраа!! ']))
    print(cn.normalize_text('стороны Заключили (ХОРОШИЙ)договор, (уррраа!!) ПРЕДМЕТ ДОГОВОРА'))
    print(cn.normalize_word('ДОГОВОР'))

  def test_normalize_doc_tokenization(self):
    doc_text = """n\n\nАкционерное общество «Газпром - Вибраниум и Криптонит» (АО «ГВК»), именуемое в \
        дальнейшем «Благотворитель», в лице заместителя генерального директора по персоналу и \
        организационному развитию Неизвестного И.И., действующего на основании на основании Доверенности № Д-17 от 29.01.2018г, \
        с одной стороны, и Фонд поддержки социальных инициатив «Интерстеларные пущи», именуемый в дальнейшем «Благополучатель», \
        в лице Генерального директора ____________________действующего на основании Устава, с другой стороны, \
        именуемые совместно «Стороны», а по отдельности «Сторона», заключили настоящий Договор о нижеследующем:
        """
    doc = LegalDocument(doc_text)
    doc.parse()
    self.assertEqual(doc.tokens_map.text.lower(), doc.tokens_map_norm.text.lower())

    for i in range(len(doc.tokens)):
      self.assertEqual(doc.tokens[i].lower(), doc.tokens_cc[i].lower())

  def test_normalize_basics(self):
    cn = CaseNormalizer()
    tm = TextMap('стороны Заключили (ХОРОШИЙ)договор, (уррраа!!) ПРЕДМЕТ ДОГОВОРА')

    tm2 = cn.normalize_tokens_map_case(tm)

    self.assertEqual(tm.map, tm2.map)
    self.assertEqual(tm2[1], 'заключили')
    self.assertEqual(tm2[12], 'Предмет')

    for i in range(len(tm)):
      self.assertEqual(tm2[i].lower(), tm[i].lower())


class TestTextNormalization(unittest.TestCase):

  def test_normalize_company_name_2(self):

    _, b = normalize_company_name('«Новые перспективы»')
    self.assertEqual('Новые перспективы', b)

    _, b = normalize_company_name('')
    self.assertEqual('', b)

    _, b = normalize_company_name(' обслуживания «Новые перспективы')
    self.assertEqual('обслуживания «Новые перспективы»', b)

    # _, b = normalize_company_name('«социального обслуживания «Новые перспективы» и «Старые тоже»')
    # self.assertEqual('социального обслуживания «Новые перспективы» и «Старые тоже»', b)

    _, b = normalize_company_name(' обслуживания «Новые» перспективы')
    self.assertEqual('обслуживания «Новые» перспективы', b)

  def test_normalize_company_name(self):

    a, b = normalize_company_name('«Институт повышения квалификации «МногоИкал')
    self.assertEqual('', a)
    self.assertEqual('Институт повышения квалификации «МногоИкал»', b)

    a, b = normalize_company_name('«Дворец культуры «Бескулька»')
    self.assertEqual('', a)
    self.assertEqual('Дворец культуры «Бескулька»', b)

    a, b = normalize_company_name('Многофункциональный комплекс «Лахта центр»')
    self.assertEqual('', a)
    self.assertEqual('Многофункциональный комплекс «Лахта центр»', b)

    a, b = normalize_company_name('ООО «Газпромнефть Марин Бункер»')
    self.assertEqual('ООО', a)
    self.assertEqual('Газпромнефть Марин Бункер', b)

    a, b = normalize_company_name('ООО «Газпромнефть «Марин Бункер»')
    self.assertEqual('ООО', a)
    self.assertEqual('Газпромнефть «Марин Бункер»', b)

    a, b = normalize_company_name('НИС а.о. Нови Сад')
    self.assertEqual('НИС а.о.', a)
    self.assertEqual('Нови Сад', b)

    a, b = normalize_company_name("ООО \"Газпромнефть-Трейд  Оренбург\"")
    self.assertEqual('ООО', a)
    self.assertEqual('Газпромнефть-Трейд Оренбург', b)

    a, b = normalize_company_name("ООО \"Газпромнефть - Трейд  Оренбург\"")
    self.assertEqual('ООО', a)
    self.assertEqual('Газпромнефть-Трейд Оренбург', b)

    a, b = normalize_company_name("АО «Газпромнефть – Аэро»")
    self.assertEqual('АО', a)
    self.assertEqual('Газпромнефть-Аэро', b)

    a, b = normalize_company_name("АО «Газпромнефть–Сахалин»")
    self.assertEqual('АО', a)
    self.assertEqual('Газпромнефть-Сахалин', b)

  def _testNorm(self, a, b):
    _norm = normalize_text(a, replacements_regex)
    # _norm2 = normalize_text(_norm, replacements_regex)
    # if not _norm == b:
    #   self.fail('\n' + _norm + ' <> \n' + b)

    self.assertEqual(b, _norm)
    # test idempotence
    # self.assertEqual(_norm2, b)

  def test_normalize_doc_1(self):
    doc_text = "«Газпром - Вибраниум и Криптонит» (АО «ГВК»)"
    doc = LegalDocument(doc_text)
    doc.parse()
    self.assertEqual(doc.text, "«Газпром - Вибраниум и Криптонит» (АО «ГВК»)")

  def test_normalize_slash(self):
    doc_text = 'с передачей в аренду/субаренду недвижимого'
    doc = LegalDocument(doc_text).parse()
    self.assertEqual('с передачей в аренду / субаренду недвижимого', doc.text)

  def test_normalize_double_quotes(self):

    tt = "''Газпром''"
    self._testNorm(tt, '"Газпром"')

  def test_normalize_doc_double_quotes(self):

    tt = "''Газпром''"
    doc = LegalDocument(tt)
    doc.parse()
    self.assertEqual('"Газпром"', doc.text)

  def test_dot_in_numbers(self):
    self._testNorm('Сумма договора не должна превышать 500 000 (пятьсот тысяч) рублей.',
                   'Сумма договора не должна превышать 500000 (пятьсот тысяч) рублей.')

    self._testNorm('Сумма договора не должна превышать 50 000 (пятьсот тысяч) рублей.',
                   'Сумма договора не должна превышать 50000 (пятьсот тысяч) рублей.')

    self._testNorm('Сумма договора не должна превышать 50 000,00 (пятьсот тысяч) рублей.',
                   'Сумма договора не должна превышать 50000,00 (пятьсот тысяч) рублей.')

    self._testNorm(
      'Стоимость оборудования 80 000,00 (восемьдесят тысяч рублей рублей 00 копеек) рублей, НДС не облагается.',
      'Стоимость оборудования 80000,00 (восемьдесят тысяч рублей рублей 00 копеек) рублей, НДС не облагается.')

    self._testNorm(
      ' Общество с ограниченной ответственностью «Газпромнефть-Сахалин» (ООО «Газпромнефть-Сахалин»), именуемое в дальнейшем «Заказчик», в лице генерального директора Коробкова Александра Николаевича, действующего на основании Устава, с одной стороны, и',
      ' Общество с ограниченной ответственностью «Газпромнефть-Сахалин» (ООО «Газпромнефть-Сахалин»), именуемое в дальнейшем «Заказчик», в лице генерального директора Коробкова Александра Николаевича, действующего на основании Устава, с одной стороны, и')

    self._testNorm('3.000 (Три тысячи) рублей 00 коп.,', '3000 (Три тысячи) рублей 00 копеек,')

    self._testNorm('составит 32.000 (Тридцать две тысячи)', 'составит 32000 (Тридцать две тысячи)')

    self._testNorm('составит 32 000 (Тридцать две тысячи)', 'составит 32000 (Тридцать две тысячи)')

    self._testNorm('42 000 (Тридцать две тысячи)', '42000 (Тридцать две тысячи)')

    self._testNorm('12 042 000 (скокато миллионов)', '12042000 (скокато миллионов)')

    self._testNorm('составит 32000', 'составит 32000')
    self._testNorm('составит 32.00', 'составит 32.00')

    self._testNorm('\x07составит', '\nсоставит')

    self._testNorm('настоящим договором в сумме 5 000 (Пять тысяч) рублей',
                   'настоящим договором в сумме 5000 (Пять тысяч) рублей')

    self._testNorm(
      '«Базовый курс » - 3.000 (Три тысячи) рублей 00 коп., - 2.000 (Две тысячи)',
      '«Базовый курс» - 3000 (Три тысячи) рублей 00 копеек, - 2000 (Две тысячи)')

  def test_normalize_numbered(self):
    self._testNorm(
      '1.этилен мама, этилен!',
      '1. этилен мама, этилен!')

  def test_normalize_numbered_1(self):
    self._testNorm(
      '2.1.этилен мама, этилен!',
      '2.1. этилен мама, этилен!')

  def test_normalize_numbered_2(self):
    self._testNorm(
      '2.01.этилен мама, этилен!',
      '2.01. этилен мама, этилен!')

  def test_normalize_caption(self):
    self._testNorm(
      'У С Т А В Акционерного общества «ГПН-ГПН »',
      'УСТАВ Акционерного общества «ГПН-ГПН»')

    self._testNorm(
      'У С Т А В',
      'УСТАВ')

  def test_normalize_numbered_3(self):
    self._testNorm(
      '112.01.этилен мама, этилен!',
      '112.01. этилен мама, этилен!')

  def test_normalize_numbered_4(self):
    self._testNorm(
      '112.01.8.Втилен мама!',
      '112.01.8. Втилен мама!')

  def test_normalize_numbered_5(self):
    self._testNorm(
      '2.01.Urengoy',
      '2.01. Urengoy')

  def testSpace1(self):
    self._testNorm('с ограниченной ответственностью « Ч» в лице',
                   'с ограниченной ответственностью «Ч» в лице')
    self._testNorm('с ограниченной ответственностью «Ч » в лице',
                   'с ограниченной ответственностью «Ч» в лице')

    self._testNorm('в г.Урюпинск тлень', 'в г. Урюпинск тлень')
    self._testNorm('в 2019г. в г.Урюпинск пельмень', 'в 2019 год в г. Урюпинск пельмень')
    self._testNorm('в 2019  г. в г.Урюпинск отчаяние', 'в 2019 год в г. Урюпинск отчаяние')
    self._testNorm('в 2019\n  г. в г.Урюпинск снег', 'в 2019 год в г. Урюпинск снег')
    self._testNorm('в 2019  г. в г.  Урюпинск снег', 'в 2019 год в г. Урюпинск снег')
    self._testNorm('в 19г. в г.  Урюпинск снег', 'в 19 г. в г. Урюпинск снег')

    self._testNorm('в г. Урюпинск снег 15(!!!) дюймов', 'в г. Урюпинск снег 15 (!!!) дюймов')

    self._testNorm('не позднее20 дней', 'не позднее 20 дней')

  def testSpace2(self):
    self._testNorm('Предложение \'\' Предложение', 'Предложение " Предложение')
    self._testNorm('Предложение " Предложение', 'Предложение " Предложение')

    self._testNorm('Предложение . Предложение', 'Предложение. Предложение')
    self._testNorm('Предложение.Предложение', 'Предложение. Предложение')
    self._testNorm('Предложение  . Предложение.', 'Предложение. Предложение.')
    self._testNorm('Предложение  , в котором', 'Предложение, в котором')
    self._testNorm('Предложение  \n\n, в котором', 'Предложение, в котором')
    self._testNorm('Абзац 1. \n\n Абзац 2. \n\n', 'Абзац 1. \n\n Абзац 2. \n\n')
    self._testNorm('Предложение  . Предложение. .25', 'Предложение. Предложение. 0.25')

    self._testNorm('пункт 2.12', 'пункт 2.12')

  def test_normalize_legal_entity_type_long(self):
    for t in legal_entity_types:
      print(t, legal_entity_types[t])

      long_, _, _ = normalize_legal_entity_type(t)
      self.assertEqual(t, long_)

  def test_normalize_legal_entity_type_short(self):
    for t in legal_entity_types:
      print(t, legal_entity_types[t])

      _, short_, _ = normalize_legal_entity_type(t)
      self.assertEqual(legal_entity_types[t], short_)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
