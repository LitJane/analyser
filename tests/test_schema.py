#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import json
import unittest

from bson import ObjectId
from jsonschema import ValidationError
from pymongo import MongoClient

import analyser
from analyser.attributes import to_json, convert_one
from analyser.ml_tools import SemanticTagBase
from analyser.runner import schema_validator
from analyser.schemas import CharterSchema, CharterStructuralLevel, Competence, ContractPrice, OrgItem, \
  Schema2LegacyListConverter, AgendaItemContract
from analyser.schemas import document_schemas
from analyser.structures import OrgStructuralLevel, ContractSubject
from integration.db import get_mongodb_connection
from tests.test_utilits import NO_DB_ERR_MSG, NO_DB


class TestSchema(unittest.TestCase):

  @unittest.skipIf(NO_DB, NO_DB_ERR_MSG)
  def test_insert_schema_to_db_json(self):
    db = get_mongodb_connection()
    collection_schemas = db['schemas']

    json_str = json.dumps(document_schemas, indent=4, ensure_ascii=False)
    print(json_str)
    print(type(json_str))
    key = f"documents_schema_{analyser.__version__}"
    collection_schemas.delete_many({"_id": key})
    collection_schemas.insert_one({"_id": key, 'json': json_str, "version": analyser.__version__})

  @unittest.skipIf(NO_DB, NO_DB_ERR_MSG)
  def test_read_schema_from_db(self):
    db = get_mongodb_connection()
    collection_schemas = db['schemas']
    key = f"documents_schema_{analyser.__version__}"
    a = collection_schemas.find_one({"_id": key})['json']
    db_document_schemas = json.loads(a)
    print(a)
    print(type(db_document_schemas))

    wrong_tree = {
      "contract": {
        "date": {
          "_ovalue": "2017-06-13T00:00:00.000Z",
          "span": [14, 17],
          "span_map": "words",
          "confidence": 1
        },
      }
    }

    with self.assertRaises(ValidationError) as context:
      schema_validator.validate(wrong_tree)

    self.assertIsNotNone(context.exception)
    print(context.exception)

  @unittest.skip
  def test_migrate_single_charter(self):
    _db_client = MongoClient(f'mongodb://192.168.10.36:27017/')
    _db_client.server_info()

    db = _db_client['gpn']

    documents_collection = db['documents']

    doc = documents_collection.find_one({"_id": ObjectId('5e4b9cd89a67394138e2089e')}, projection={
      '_id': True,
      'analysis.attributes': True,
      'user.attributes': True,
      'parse.documentType': True})

    convert_one(db, doc)

  def test_enum_to_json(self):
    d = {
      "some": OrgStructuralLevel.CEO
    }

    a, b = to_json(d)

    print(a)
    print(a)

  def test_convert_to_legasy_list(self):
    cs = CharterSchema()

    cs.org = OrgItem()
    cs.org.name = SemanticTagBase()
    cs.org.type = SemanticTagBase()
    cs.org.alias = SemanticTagBase()

    cs.date = SemanticTagBase()

    structural_level = CharterStructuralLevel()
    structural_level.value = OrgStructuralLevel.BoardOfCompany
    structural_level.span = (1, 2)

    structural_level1 = CharterStructuralLevel()
    structural_level1.value = OrgStructuralLevel.AllMembers
    structural_level1.span = (1, 2)

    structural_level.confidence = 0.777
    cs.structural_levels.append(structural_level)
    cs.structural_levels.append(structural_level1)

    comp = Competence()
    comp.confidence = 0.22
    comp.span = (2, 2)
    comp.value = ContractSubject.Charity
    structural_level.competences.append(comp)

    cp1 = ContractPrice()
    cp1.sign = SemanticTagBase()
    cp1.sign.value = -1

    cp2 = ContractPrice()
    cp2.sign = SemanticTagBase()
    cp2.sign.value = 1
    cp2.amount = SemanticTagBase()
    cp2.amount.value = 1
    cp2.currency = SemanticTagBase()
    cp2.currency.value = "USD"

    cp3 = ContractPrice()
    cp3.amount = SemanticTagBase()

    comp.constraints = [cp1, cp2, cp3]

    converter = Schema2LegacyListConverter()
    dest = {}
    converter.schema2list(dest, cs)
    for k, v in dest.items():
      print(f"[{k}]", v)

    self.assertTrue('BoardOfCompany/Charity/constraint-max-1' in dest)
    self.assertTrue('org-1-name' in dest)
    self.assertTrue('org-1-type' in dest)
    self.assertTrue('org-1-alias' in dest)
    # print(dest)

  def test_date_wrong_2(self):
    tree = {
      "contract": {
        "date": {

          "value": "wrong date",
          "span": [14, 17],
          "span_map": "words",
          "confidence": 1
        },
      }
    }

    with self.assertRaises(Exception) as context:
      schema_validator.validate(tree)
      print(context)

    self.assertIsNotNone(context.exception)
    print(context.exception)

  def test_date_wrong_3(self):
    tree = {
      "contract": {
        "date": {
          "_ovalue": "2017-06-13T00:00:00.000Z",
          "span": [14, 17],
          "span_map": "words",
          "confidence": 1
        },
      }
    }

    with self.assertRaises(ValidationError) as context:
      schema_validator.validate(tree)

    self.assertIsNotNone(context.exception)
    print(context.exception)

  def test_date_wrong(self):
    tree = {
      "contract": {
        "date": {
          "value": "2017-06-13T00:00:00.000Z",
          "span_map": "words",
          "confidence": 1
        },
      }
    }

    with self.assertRaises(ValidationError) as context:
      schema_validator.validate(tree)

    self.assertIsNotNone(context.exception)

  @staticmethod
  def test_date_correct():
    tree = {
      "contract": {
        "date": {
          "value": "2017-06-13T00:00:00.000Z",
          "span": [14, 17],
          "span_map": "words",
          "confidence": 1
        },
      }
    }

    schema_validator.validate(tree)

  @staticmethod
  def test_value_correct():
    tree = {
      "contract": {
        "price": {
          "amount": {
            "value": 2000,
            "span": [14, 17],
          },
          "sign": {
            "value": -1,
            "span": [14, 17],
          },
          "vat": {
            "value": 20,
            "span": [17, 20],
          },
          "currency": {
            "value": "RUB",
            "span": [14, 17],
          },
          "span": [14, 17]
        },
      }
    }

    schema_validator.validate(tree)

  def test_value_missing_fields(self):
    tree = {
      "contract": {
        "price": {

          "sign": {
            "value": -1,
            "span": [14, 17],
          },
          "currency": {
            "value": "RUB",
            "span": [14, 17],
          },
          "span": [14, 17]
        },
      }
    }
    with self.assertRaises(ValidationError) as context:
      schema_validator.validate(tree)
    self.assertIsNotNone(context.exception)

  @staticmethod
  def test_subject_correct():
    tree = {
      "contract": {
        "subject": {
          "value": 'Charity',
          "span": [14, 17],

        },
      }
    }

    schema_validator.validate(tree)

  def test_subject_wrong(self):
    tree = {
      "contract": {
        "subject": {
          "value": 'Charity WRONG',
          "span": [14, 17],

        },
      }
    }

    with self.assertRaises(ValidationError) as context:
      schema_validator.validate(tree)
    self.assertIsNotNone(context.exception)

  @staticmethod
  def test_org_correct():
    tree = {
      "contract": {"orgs": [{
        "name": {
          "value": "ГПН",
          "span": [30, 31],
          "span_map": "words",
          "confidence": 0.8
        },
        "type": {
          "value": "Акционерное общество",
          "span": [27, 29],
          "span_map": "words",
          "confidence": 0.8
        }
      }]}
    }

    schema_validator.validate(tree)

  def test_org_wrong(self):
    tree = {
      "contract": {
        "orgs": [{
          "name": {
            "Xvalue": "ГПН",
            "span": [30, 31],
            "span_map": "words",
            "confidence": 0.8
          },
          "type": {
            "Xvalue": "Акционерное общество",
            "span": [27, 29],
            "span_map": "words",
            "confidence": 0.8
          }
        }]}
    }

    with self.assertRaises(ValidationError) as context:
      schema_validator.validate(tree)

    self.assertIsNotNone(context.exception)

  def test_AgendaItemContract(self):
    aic = AgendaItemContract()
    aic.date = SemanticTagBase()
    aic.date.span = (10, 20)

    self.assertEqual(aic.span, (10, 20))

    aic.price = SemanticTagBase()
    aic.price.span = (15, 30)
    self.assertEqual(aic.span, (10, 30))

    aic.number = SemanticTagBase()
    aic.number.span = (50, 100)
    self.assertEqual(aic.span, (10, 100))

    print(aic.span)

    aic.orgs.append(OrgItem())
    aic.orgs[0].type = SemanticTagBase()
    aic.orgs[0].type.span = (110, 120)
    self.assertEqual(aic.span, (10, 120))


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
