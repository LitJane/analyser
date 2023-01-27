#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

# os.environ['GPN_DB_HOST']='192.168.10.36'

import unittest

from bson import ObjectId

from analyser.finalizer import get_doc_by_id, get_audit_by_id
from analyser.log import logger
from analyser.parsing import AuditContext
from analyser.persistence import DbJsonDoc
from analyser.runner import BaseProcessor, document_processors, CONTRACT, PROTOCOL, CHARTER
from integration.db import get_mongodb_connection

# @unittest.skip
class AnalyzerTestCase(unittest.TestCase):
  @unittest.skip
  def test_analyse_acontract(self):
    # {_id:ObjectId('5de8a3fd1b3453848224a9d5')}
    doc = get_doc_by_id(ObjectId('60b7a509061c76d775454b51'))
    # _db_client = MongoClient(f'mongodb://192.168.10.36:27017/')
    # _db_client.server_info()

    # db = _db_client['gpn']

    # documents_collection = db['documents']

    # doc = documents_collection.find_one({"_id": ObjectId('5fdb213f542ce403c92b4530')} )
    # audit = db['audits'].find_one({'_id': doc['auditId']})
    audit = get_audit_by_id(doc['auditId'])
    jdoc = DbJsonDoc(doc)
    logger.info(f'......pre-processing {jdoc._id}')
    # _audit_subsidiary: str = audit["subsidiary"]["name"]

    ctx = AuditContext(None )
    processor: BaseProcessor = document_processors[CONTRACT]
    processor.preprocess(jdoc, context=ctx)
    processor.process(jdoc, audit, ctx)
    print(jdoc)

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_analyze_contract(self):
    processor: BaseProcessor = document_processors[CONTRACT]
    doc = get_doc_by_id(ObjectId('63c92b5dfb21317f1d5e0f83'))
    if doc is None:
      raise RuntimeError("fix unit test please, doc with given UID is not in test DB")

    audit = get_audit_by_id(doc['auditId'])
    # print(audit)

    jdoc = DbJsonDoc(doc)
    logger.info(f'......pre-processing {jdoc._id}')
    ctx = AuditContext()
    ctx.audit_subsidiary_name=audit.get('subsidiary',{}).get('name')
    processor.preprocess(jdoc, context=ctx)
    processor.process(jdoc, audit, ctx)


  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_analyze_protocol(self):
    processor: BaseProcessor = document_processors[PROTOCOL]
    doc = get_doc_by_id(ObjectId('5e5de70b01c6c73c19eebd35'))
    if doc is None:
      raise RuntimeError("fix unit test please, doc with given UID is not in test DB")

    audit = get_audit_by_id(doc['auditId'])

    jdoc = DbJsonDoc(doc)
    logger.info(f'......pre-processing {jdoc._id}')
    ctx = AuditContext()
    processor.preprocess(jdoc, context=ctx)
    processor.process(jdoc, audit, ctx)

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_analyze_charter(self):
    processor: BaseProcessor = document_processors[CHARTER]
    doc = get_doc_by_id(ObjectId('60c371b7862b20b4ba55c735'))
    if doc is None:
      raise RuntimeError("fix unit test please, doc with given UID is not in test DB")

    audit = None #get_audit_by_id(doc['auditId'])

    jdoc = DbJsonDoc(doc)
    logger.info(f'......pre-processing {jdoc._id}')
    ctx = AuditContext()
    processor.preprocess(jdoc, context=ctx)
    doc = processor.process(jdoc, audit, ctx)

    # print(doc)

#


if __name__ == '__main__':
  unittest.main()
