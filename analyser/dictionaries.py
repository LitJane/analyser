import json
import os
from pathlib import Path

from pymongo import DESCENDING, ASCENDING

import analyser
from analyser.charter_parser import CharterParser
from analyser.schemas import document_schemas
from analyser.structures import OrgStructuralLevel, ContractSubject, contract_subjects, \
  legal_entity_types
from gpn.gpn import subsidiaries
from integration.db import get_mongodb_connection

integration_path = Path(analyser.__file__).parent.parent / 'integration' / 'classifier'  # .parent/'integration/classifier'

with open(integration_path / 'practices.json', encoding='utf-8') as practice_json_file:
  all_labels = json.load(practice_json_file)

labels = list(map(lambda item: item['label'], filter(lambda item: item['auto-classified'], all_labels)))

label2id = {item['label']: item['_id'] for item in all_labels}


def contract_subject_as_db_json():
  for cs in ContractSubject:
    item = {
      '_id': cs.name,
      'number': cs.value,
      'alias': cs.display_string,
      'supportedInContracts': cs in contract_subjects,
      'supportedInCharters': cs in CharterParser.strs_subjects_patterns.keys()
    }
    yield item


def legal_entity_types_as_db_json():
  for k in legal_entity_types.keys():
    yield {'_id': k, 'alias': legal_entity_types[k]}


def insert_schemas_to_db(db):
  collection_schemas = db['schemas']

  json_str = json.dumps(document_schemas, indent=4)
  # print(json_str)
  # print(type(json_str))
  key = f"documents_schema_{analyser.__version__}"
  collection_schemas.delete_many({"_id": key})
  collection_schemas.insert_one({"_id": key, 'json': json_str, "version": analyser.__version__})


def update_db_dictionaries():
  db = get_mongodb_connection()

  insert_schemas_to_db(db)

  if os.environ.get("GPN_CSGK_WSDL") is None:
    coll = db["subsidiaries"]
    coll.delete_many({})
    coll.insert_many(subsidiaries)

  coll = db["orgStructuralLevel"]
  coll.delete_many({})
  coll.insert_many(OrgStructuralLevel.as_db_json())

  coll = db["legalEntityTypes"]
  coll.delete_many({})
  coll.insert_many(legal_entity_types_as_db_json())

  coll = db["contractSubjects"]
  coll.delete_many({})
  coll.insert_many(contract_subject_as_db_json())

  coll = db["analyser"]
  coll.delete_many({})
  coll.insert_one({'version': analyser.__version__})

  coll = db['practices']
  coll.delete_many({})
  coll.insert_many(all_labels)
  coll.create_index('tessa_id')

  # indexing
  print('creating db indices')
  coll = db["documents"]

  resp = coll.create_index([("analysis.analyze_timestamp", DESCENDING)])
  print("index response:", resp)
  resp = coll.create_index([("user.updateDate", DESCENDING)])
  print("index response:", resp)
  resp = coll.create_index([("analysis.attributes.date.value", DESCENDING)])
  print("index response:", resp)

  coll = db["documents"]
  sorting = [('analysis.analyze_timestamp', ASCENDING), ('user.updateDate', ASCENDING)]
  resp = coll.create_index(sorting)
  print("index response:", resp)

  coll = db['audits']
  coll.create_index('email_sent')
  coll.create_index('additionalFields.external_source')


if __name__ == '__main__':
  update_db_dictionaries()
