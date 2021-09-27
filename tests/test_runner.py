#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

import pymongo
from bson import ObjectId

from analyser import finalizer
from analyser.finalizer import get_doc_by_id, get_audit_by_id
from analyser.log import logger
from analyser.parsing import AuditContext
from analyser.persistence import DbJsonDoc
from analyser.runner import Runner, get_audits, get_docs_by_audit_id, document_processors, save_analysis, \
  contract_processor
from integration.db import get_mongodb_connection

SKIP_TF = True


def get_runner_instance_no_embedder() -> Runner:
  if TestRunner.default_no_tf_instance is None:
    TestRunner.default_no_tf_instance = Runner(init_embedder=False)
  return TestRunner.default_no_tf_instance


@unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
class TestRunner(unittest.TestCase):
  default_no_tf_instance: Runner = None

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_is_valid(self):
    doc = get_doc_by_id(ObjectId('5fb3d79f78df3635f5441d31'))
    if doc is None:
      raise RuntimeError("fix unit test please, use valid OID")

    audit = get_audit_by_id(doc['auditId'])
    jdoc = DbJsonDoc(doc)
    contract_processor.is_valid(audit, jdoc)
    # is_va

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_get_audits(self):
    aa = get_audits()
    for a in aa:
      print(a['_id'])

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_get_docs_by_audit_id(self):
    audits = get_audits()
    if len(audits) == 0:
      logger.warning('no audits')
      return

    audit_id = audits[0]['_id']

    docs = get_docs_by_audit_id(audit_id, kind='PROTOCOL')
    for a in docs:
      print(a['_id'], a['filename'])

  def _get_doc_from_db(self, kind):
    audits = get_mongodb_connection()['audits'].find().sort([("createDate", pymongo.ASCENDING)]).limit(1)
    for audit in audits:
      doc_ids = get_docs_by_audit_id(audit['_id'], kind=kind, states=[15], id_only=True)
      if len(doc_ids) > 0:
        print(doc_ids[0])
        doc = finalizer.get_doc_by_id(doc_ids[0])
        # jdoc = DbJsonDoc(doc)
        yield doc

  def _preprocess_single_doc(self, kind):
    for doc in self._get_doc_from_db(kind):
      d = DbJsonDoc(doc)
      processor = document_processors.get(kind)
      processor.preprocess(d, AuditContext())

  # @unittest.skipIf(SKIP_TF, "requires TF")

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_preprocess_single_protocol(self):
    self._preprocess_single_doc('PROTOCOL')

  @unittest.skipIf(get_mongodb_connection() is None is None, "requires mongo")
  def test_preprocess_single_contract(self):
    self._preprocess_single_doc('CONTRACT')

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_process_contracts_phase_1(self):
    # runner = Runner.get_instance()

    audits = get_audits()
    if len(audits) == 0:
      logger.warning('no audits')
      return

    audit_id = audits[0]['_id']

    docs = get_docs_by_audit_id(audit_id, kind='CONTRACT')
    processor = document_processors.get('CONTRACT')
    for _doc in docs:
      jdoc = DbJsonDoc(_doc)
      processor.preprocess(jdoc, AuditContext())

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_process_charters_phase_1(self):
    audits = get_audits()
    if len(audits) == 0:
      logger.warning('no audits')
      return

    audit_id = audits[0]['_id']
    docs: [dict] = get_docs_by_audit_id(audit_id, kind='CHARTER')
    processor = document_processors.get('CHARTER')
    for _doc in docs:
      jdoc = DbJsonDoc(_doc)
      processor.preprocess(jdoc, AuditContext())

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_process_protocols_phase_1(self):
    runner = get_runner_instance_no_embedder()

    for audit in get_audits():
      audit_id = audit['_id']
      docs = get_docs_by_audit_id(audit_id, kind='PROTOCOL')

      for doc in docs:
        # charter = runner.make_legal_doc(doc)

        jdoc = DbJsonDoc(doc)
        legal_doc = jdoc.asLegalDoc()

        runner.protocol_parser.find_org_date_number(legal_doc, AuditContext())
        save_analysis(jdoc, legal_doc, -1)

  # if get_mongodb_connection() is not None:
  unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
# else:
#   warnings.warn('mongo connection is not available')
