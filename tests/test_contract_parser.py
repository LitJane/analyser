#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import os
import pickle
import unittest
import warnings

from contract_parser import ContractAnlysingContext, ContractDocument
from contract_patterns import ContractPatternFactory
from documents import TextMap
from legal_docs import LegalDocument
from ml_tools import SemanticTag
from structures import ContractTags


class TestContractParser(unittest.TestCase):

  def get_doc(self) -> (ContractDocument, ContractPatternFactory):
    pth = os.path.dirname(__file__)
    with open(pth + '/2. Договор по благ-ти Радуга.docx.pickle', 'rb') as handle:
      doc = pickle.load(handle)

    with open(pth + '/contract_pattern_factory.pickle', 'rb') as handle:
      factory = pickle.load(handle)

    self.assertEqual(1024, doc.embeddings.shape[-1])

    return doc, factory

  def _get_doc_factory_ctx(self):
    doc, factory = self.get_doc()

    ctx = ContractAnlysingContext(embedder={}, renderer=None, pattern_factory=factory)
    ctx.verbosity_level = 3
    ctx.sections_finder.find_sections(doc, ctx.pattern_factory, ctx.pattern_factory.headlines,
                                      headline_patterns_prefix='headline.')
    return doc, factory, ctx

  def test_contract_analyze(self):
    doc, factory, ctx = self._get_doc_factory_ctx()

    ctx.analyze_contract_doc(doc)
    tags: [SemanticTag] = doc.get_tags()

    _tag = SemanticTag.find_by_kind(tags, ContractTags.Value.display_string)
    quote = doc.tokens_map.text_range(_tag.span)
    self.assertEqual('80000,00', quote)

    _tag = SemanticTag.find_by_kind(tags, ContractTags.Currency.display_string)
    quote = doc.tokens_map.text_range(_tag.span)
    self.assertEqual('рублей', quote)

  def print_semantic_tag(self, tag: SemanticTag, map: TextMap):
    print('print_semantic_tag:', tag, f"[{map.text_range(tag.span)}]", tag.parent)

  def test_find_contract_value(self):
    doc, factory = self.get_doc()

    ctx = ContractAnlysingContext(embedder={}, renderer=None, pattern_factory=factory)
    ctx.verbosity_level = 3
    ctx.sections_finder.find_sections(doc, ctx.pattern_factory, ctx.pattern_factory.headlines,
                                      headline_patterns_prefix='headline.')
    # ----------------------------------------
    values = ctx.find_contract_value_NEW(doc)
    # ----------------------------------------

    self.assertEqual(1, len(values))
    v = values[0]

    value = SemanticTag.find_by_kind(v, ContractTags.Value.display_string)
    sign = SemanticTag.find_by_kind(v, ContractTags.Sign.display_string)
    currency = SemanticTag.find_by_kind(v, ContractTags.Currency.display_string)

    self.print_semantic_tag(sign, doc.tokens_map)
    self.print_semantic_tag(value, doc.tokens_map)
    self.print_semantic_tag(currency, doc.tokens_map)

    self.assertEqual(0, sign.value)
    self.assertEqual(80000, value.value)
    self.assertEqual('RUB', currency.value)

  def test_find_contract_subject(self):
    warnings.warn("use find_contract_subject_region", DeprecationWarning)
    doc, factory, ctx = self._get_doc_factory_ctx()
    # ----------------------------------------
    subject: SemanticTag = ctx.find_contract_subject_region(doc)
    # ----------------------------------------

    print(subject)
    self.assertEqual('Charity', subject.value)

  def test_find_contract_sections(self):

    doc, factory, ctx = self._get_doc_factory_ctx()
    # ----------------------------------------

    ctx.sections_finder.find_sections(doc, ctx.pattern_factory, ctx.pattern_factory.headlines,
                                      headline_patterns_prefix='headline.')
    # ----------------------------------------
    print("SECTION:")
    for section in doc.sections.keys():
      print(section, doc.sections[section].confidence, doc.sections[section].header.strip())
      self.assertGreater(doc.sections[section].confidence, 0.7, doc.sections[section].header.strip())

    print("PARAGs:")
    for p in doc.paragraphs:
      print(p.header.value)

    self.assertIn('subj', doc.sections)
    self.assertIn('price.', doc.sections)
    self.assertIn('pricecond', doc.sections)

    self.assertEqual('1. ПРЕДМЕТ ДОГОВОРА.', doc.sections['subj'].header.strip())

  def test_find_contract_subject_region_in_subj_section(self):
    doc, factory, ctx = self._get_doc_factory_ctx()

    subj_section = doc.sections['subj']
    section: LegalDocument = subj_section.body
    # ----------------------------------------
    result = ctx.find_contract_subject_regions(section)
    # ---------------------

    self.print_semantic_tag(result, doc.tokens_map)
    expected = """1.1 Благотворитель оплачивает следующий счет, выставленный на Благополучателя: \n1.1.1. Счет № 115 на приобретение спортивного оборудования (теннисный стол, рукоход с перекладинами, шведская стенка). Стоимость оборудования 80000,00 (восемьдесят тысяч рублей рублей 00 копеек) рублей, НДС не облагается."""
    self.assertEqual(expected, doc.tokens_map.text_range(result.span).strip())
    self.assertEqual('Charity', result.value)

  def test_find_contract_subject_region_in_doc_head(self):
    doc, factory, ctx = self._get_doc_factory_ctx()

    section = doc.subdoc_slice(slice(0, 1500))
    denominator = 0.7

    # subj_section = doc.sections['subj']
    # section: LegalDocument = subj_section.body
    # ----------------------------------------
    result = ctx.find_contract_subject_regions(section, denominator)
    # ---------------------

    self.print_semantic_tag(result, doc.tokens_map)
    self.assertEqual(
      '1. ПРЕДМЕТ ДОГОВОРА.\n1.1 Благотворитель оплачивает следующий счет, выставленный на Благополучателя:',
      doc.tokens_map.text_range(result.span).strip())

  def test_find_contract_subject_region(self):
    doc, factory, ctx = self._get_doc_factory_ctx()

    # ----------------------------------------
    result = ctx.find_contract_subject_region(doc)
    # ---------------------

    self.print_semantic_tag(result, doc.tokens_map)
    expected = """1.1 Благотворитель оплачивает следующий счет, выставленный на Благополучателя: \n1.1.1. Счет № 115 на приобретение спортивного оборудования (теннисный стол, рукоход с перекладинами, шведская стенка). Стоимость оборудования 80000,00 (восемьдесят тысяч рублей рублей 00 копеек) рублей, НДС не облагается."""
    self.assertEqual(expected, doc.tokens_map.text_range(result.span).strip())
    self.assertEqual('Charity', result.value)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
