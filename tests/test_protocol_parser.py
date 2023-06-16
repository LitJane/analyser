#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import os
import pickle
import re
import unittest

from analyser.contract_agents import ORG_LEVELS_re
from analyser.contract_patterns import ContractPatternFactory
from analyser.legal_docs import LegalDocument
from analyser.ml_tools import SemanticTag
from analyser.parsing import AuditContext
from analyser.persistence import DbJsonDoc
from analyser.protocol_parser import find_protocol_org, find_org_structural_level, protocol_votes_re, ProtocolDocument
from analyser.runner import Runner
from analyser.structures import OrgStructuralLevel
from tests.test_utilits import load_json_sample


class TestProtocolParser(unittest.TestCase):

  def test_read_json(self):
    data = load_json_sample('protocol_1.json')
    self.assertIsNotNone(data)

  def test_protocol_processor(self):
    json_doc = load_json_sample('protocol_1.json')
    jdoc = DbJsonDoc(json_doc)
    legal_doc = jdoc.asLegalDoc()

    pp = Runner.get_instance().protocol_parser

    legal_doc: ProtocolDocument = pp.find_attributes(legal_doc, AuditContext())

    orgtags = legal_doc.org_tags
    for t in orgtags:
      print(t)

    def tag_val(name):
      tag = SemanticTag.find_by_kind(orgtags, name)
      if tag is not None:
        return tag.value

    self.assertEqual('Газпромнефть Шиппинг', tag_val('org-1-name'))
    self.assertEqual('Общество с ограниченной ответственностью', tag_val('org-1-type'))

  def get_doc(self, fn) -> (LegalDocument, ContractPatternFactory):
    pth = os.path.dirname(__file__)
    with open(os.path.join(pth, fn), 'rb') as handle:
      doc = pickle.load(handle)

    self.assertEqual(1024, doc.embeddings.shape[-1])

    return doc

  def test_load_picke(self):
    doc = self.get_doc('Протокол_СД_ 3.docx.pickle')
    doc: LegalDocument = doc

  def test_find_protocol_org_1(self):
    suff = ' ' * 1000

    txt = '''Протокол № 3/2019 Проведения итогов заочного голосования Совета директоров Общества с ограниченной ответственностью «Технологический центр «Бажен» (далее – ООО «Технологический центр «Бажен») г. Санкт-Петербург Дата составления протокола «__» _______ 2019 года
    Дата окончания приема бюллетеней для голосования членов Совета директоров «___»__________ 2019 года.
    ''' + suff
    doc = ProtocolDocument(LegalDocument(txt))
    doc.parse()
    tags = find_protocol_org(doc)
    self.assertEqual('Технологический центр «Бажен»', tags[0].value)
    self.assertEqual('Общество с ограниченной ответственностью', tags[1].value)

  def test_find_protocol_org_2(self):
    doc = self.get_doc('Протокол_СД_ 3.docx.pickle')
    doc.parse()

    tags = find_protocol_org(doc)
    self.assertEqual('Технологический центр «Бажен»', tags[0].value)
    self.assertEqual('Общество с ограниченной ответственностью', tags[1].value)

  def test_ORG_LEVELS_re(self):
    suff = ' ' * 300
    t = '''
    ПРОТОКОЛ
заседания Совета директоров ООО «Газпромнефть- Корпоративные продажи» (далее – ООО «Газпромнефть- Корпоративные продажи» или «Общество»)
Место проведения заседания:
''' + suff
    r = re.compile(ORG_LEVELS_re, re.MULTILINE | re.IGNORECASE | re.UNICODE)
    x = r.search(t)
    self.assertEqual('Совета директоров', x['org_structural_level'])

  def test_find_org_structural_level(self):
    t = '''
    ПРОТОКОЛ \
    заседания Совета директоров ООО «Газпромнефть - Внеземная Любофьи» (далее – ООО «Газпромнефть-ВНЛ» или «Общество»)\
    Место проведения заседания:
    ''' + ' ' * 900
    doc = LegalDocument(t)
    doc.parse()

    tags = list(find_org_structural_level(doc))
    self.assertEqual(OrgStructuralLevel.BoardOfDirectors.name, tags[0].value)

  def test_find_org_structural_level_2(self):
    t = '''
    ПРОТОКОЛ ночного заседания Правления общества ООО «Газпромнефть - Внеземная Любофь» (далее – ООО «Газпромнефть- ВНЛ» или «Общество»)\
    Место проведения заседания:
    ''' + ' ' * 900
    doc = LegalDocument(t)
    doc.parse()

    tags = list(find_org_structural_level(doc))
    self.assertEqual(OrgStructuralLevel.BoardOfCompany.name, tags[0].value)

  def test_find_protocol_votes(self):
    doc = self.get_doc('Протокол_СД_ 3.docx.pickle')
    x = protocol_votes_re.search(doc.text)

    print(doc.text[x.span()[0]:x.span()[1]])

  def test_find_protocol_votes_re(self):
    t = '''
Предварительно утвердить годовой отчет Общества за 2017 год.
Итоги голосования:
 «ЗА»              8;
«ПРОТИВ»        нет;
«ВОЗДЕРЖАЛСЯ»  нет.
РЕШЕНИЕ ПРИНЯТО.
Решение, принятое по первому вопросу повестки дня:
Предварительно утвердить годовой отчет Общества за 2017 год.'''

    doc = LegalDocument(t)
    doc.parse()

    x = protocol_votes_re.search(doc.text)

    match = doc.text[x.span()[0]:x.span()[1]]
    print(f'[{match}]')


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
