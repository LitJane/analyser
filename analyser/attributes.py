import json
import logging
import sys
from datetime import datetime, date
from enum import Enum

import jsonpickle
import numpy as np
from bson import json_util
from bson.objectid import ObjectId
from jsonschema import validate, FormatChecker

import analyser
from analyser.ml_tools import SemanticTagBase
from analyser.schemas import document_schemas, ProtocolSchema, OrgItem, AgendaItem, AgendaItemContract, HasOrgs, \
  ContractPrice, ContractSchema, CharterSchema, CharterStructuralLevel, Competence
from analyser.structures import OrgStructuralLevel, ContractSubject
from integration.db import get_doc_by_id
from integration.db import get_mongodb_connection

migration_logger = logging.getLogger('db_migration')

ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
migration_logger.setLevel(logging.DEBUG)

migration_logger.addHandler(ch)


class DatetimeHandler(jsonpickle.handlers.BaseHandler):
  def flatten(self, obj: datetime, data):
    return json_util.default(obj)


class NumpyFloatHandler(jsonpickle.handlers.BaseHandler):
  def flatten(self, obj: np.float, data):
    return round(obj, 6)


class AgendaItemContractHandler(jsonpickle.handlers.BaseHandler):
  def flatten(self, obj: AgendaItemContract, data):
    pickler = self.context
    data['span'] = pickler.flatten(obj.span)
    data['orgs'] = pickler.flatten(obj.orgs)
    data['date'] = pickler.flatten(obj.date)
    data['price'] = pickler.flatten(obj.price)
    data['number'] = pickler.flatten(obj.number)
    return data  # [obj.span, pickler.flatten(obj.orgs), obj.date, obj.price, obj.number]


class EnumHandler(jsonpickle.handlers.BaseHandler):
  def flatten(self, e: Enum, data):
    return e.name


jsonpickle.handlers.registry.register(datetime, DatetimeHandler)
jsonpickle.handlers.registry.register(date, DatetimeHandler)
jsonpickle.handlers.registry.register(Enum, EnumHandler, base=True)

jsonpickle.handlers.registry.register(np.float, NumpyFloatHandler)
jsonpickle.handlers.registry.register(np.float32, NumpyFloatHandler)
jsonpickle.handlers.registry.register(np.float64, NumpyFloatHandler)

jsonpickle.handlers.registry.register(AgendaItemContract, AgendaItemContractHandler)


def del_none(d):
  """
  Delete keys with the value ``None`` in a dictionary, recursively.

  This alters the input so you may wish to ``copy`` the dict first.
  """

  for key, value in list(d.items()):
    if value is None:
      del d[key]
    elif isinstance(value, dict):
      del_none(value)

    elif isinstance(value, list):
      for itm in list(value):
        if isinstance(itm, dict):
          del_none(itm)
      if len(value) == 0:
        del d[key]
  return d  # For convenience


def to_json(tree):
  json_str = jsonpickle.encode(tree, unpicklable=False, indent=4)

  j = json.loads(json_str, object_hook=json_util.object_hook)
  j = del_none(j)
  return j, json_str


def convert_org(attr_name: str,
                attr: dict,
                dest: HasOrgs):
  if attr_name.endswith("-alt-name"):
    attr_name = attr_name.replace("-alt-name", "-alt_name")

  name_parts = attr_name.split('-')
  _index = int(name_parts[1]) - 1
  org = array_set_or_get_at(dest.orgs, _index, OrgItem)

  field_name = name_parts[-1]

  if field_name in ['type', 'name', 'alias', 'alt_name']:
    copy_leaf_tag(field_name, src=attr, dest=org, attr_name=field_name)


def has_non_blanc_attr(dest, field_name: str) -> bool:
  if hasattr(dest, field_name):
    return getattr(dest, field_name) is not None
  return False


def copy_leaf_tag(field_name: str, src, dest, attr_name=None):
  if has_non_blanc_attr(dest, field_name):
    v = getattr(dest, field_name)
    # setattr(v, "warning", "ambiguity: multiple values, see 'alternatives' field")
    migration_logger.warning(f"{field_name} has multiple values")

    alternatives = []
    if not hasattr(v, "alternatives"):
      setattr(v, "alternatives", alternatives)
    else:
      alternatives = getattr(v, "alternatives")

    va = SemanticTagBase()

    copy_attr(src, va)
    alternatives.append(va)

  else:
    v = SemanticTagBase()
    setattr(dest, field_name, v)

    copy_attr(src, v)


def map_tag(src, dest: SemanticTagBase = None) -> SemanticTagBase:
  if dest is None:
    dest = SemanticTagBase()

  _fields = ['value', 'span', 'confidence']
  for f in _fields:
    setattr(dest, f, src.get(f))

  return dest


def getput_node(subtree, key_name, defaultValue):
  _node = subtree.get(key_name, defaultValue)
  subtree[key_name] = _node  # put_it
  return _node


def array_set_or_get_at(arr, index, builder):
  for _i in range(len(arr), index + 1):
    arr.append(None)

  preexistent = arr[index]
  if preexistent is None:
    preexistent = builder()
    arr[index] = preexistent
  return preexistent


def _find_by_value(arr: [SemanticTagBase], value):
  for c in arr:
    if c.value == value:
      return c


def convert_competence(path_s: [str], attr, competence_node: Competence):
  # charter
  constraint = path_s[0].split('-')  # example: 'constraint-min-2' or just `constraint`
  constraint_margin_index = 0
  if len(constraint) == 3:
    constraint_margin_index = int(constraint[2]) - 1

  # extending array
  margin_node = array_set_or_get_at(competence_node.constraints, constraint_margin_index, ContractPrice)

  if len(path_s) == 1:
    copy_attr(attr, dest=margin_node)
  else:
    handle_sign_value_currency(path_s[1:], attr, margin_node)


def handle_sign_value_currency(path: [str], v, dest: ContractPrice):
  if len(path) == 1:
    _pname = path[0]

    if path[0] in ["sign", "currency"]:
      copy_leaf_tag(_pname, v, dest)
    if path[0] == "value":
      copy_leaf_tag('amount', v, dest)


def convert_constraints(path_s: [str], attr, structural_level_node: CharterStructuralLevel):
  subj_name_parts = path_s[0].split('-')
  subj_name = subj_name_parts[0]
  subj = ContractSubject[subj_name]
  subj_index = 0
  if len(subj_name_parts) == 2:
    subj_index = int(subj_name_parts[1])

  if subj_index > 0:
    migration_logger.error(f"doubled {path_s} {attr.get('kind')}")

  subject_node = _find_by_value(structural_level_node.competences, subj)  # "Deal", for example
  if subject_node is None:
    subject_node = Competence(value=subj)
    structural_level_node.competences.append(subject_node)

  if len(path_s) == 1:
    copy_attr(attr, dest=subject_node, skip_value=True)
  else:
    convert_competence(path_s[1:], attr, subject_node)
  # ------------


def remove_empty_from_list(lst):
  return [v for v in lst if v is not None]


def clean_up_tree(tree: CharterSchema):
  for s_node in tree.structural_levels:
    for subj_node in s_node.competences:
      subj_node.constraints = remove_empty_from_list(subj_node.constraints)


def index_of_key(s: str) -> (str, int):
  n_i = s.split("-")
  _idx = 0
  if len(n_i) > 1:
    try:
      _idx = int(n_i[-1]) - 1
    except ValueError:
      return s, 0
  return n_i[0], _idx


def convert_agenda_item(path, attr: {}, _item_node: AgendaItem):
  if _item_node.contract is None:
    _item_node.contract = AgendaItemContract()

  c_node: AgendaItemContract = _item_node.contract

  attr_name = path[0]  # 'contract_agent_org-1-type'
  attr_name_parts = attr_name.split("-")
  attr_base_name = attr_name_parts[0]  # 'contract_agent_org'
  if "contract_agent_org" == attr_base_name:
    convert_org(attr_name, attr=attr, dest=c_node)

  if len(path) > 1 and "sign_value_currency" == path[1]:
    convert_sign_value_currency(path[1:], attr, c_node)

  if attr_base_name in ['date', 'number']:
    copy_leaf_tag(attr_base_name, src=attr, dest=c_node)


def copy_attr(src, dest: SemanticTagBase, skip_value=False) -> SemanticTagBase:
  _list = ['span', 'span_map', 'confidence', "value"]
  if skip_value:
    _list = ['span', 'span_map', 'confidence']
  for key in _list:
    setattr(dest, key, src.get(key))

  return dest


def map_org(attr_name: str, v, dest: OrgItem) -> OrgItem:
  name_parts = attr_name.split('-')
  _index = int(name_parts[1]) - 1
  _field = name_parts[-1]
  if _field in ['name', 'type']:
    setattr(dest, _field, map_tag(v))

  return dest


def list_tags(attrs):
  for path, v in attrs.items():
    key_s: [] = path.split('/')

    if v.get('span', None):
      yield key_s, v


def convert_contract_db_attributes_to_tree(attrs) -> ContractSchema:
  tree = ContractSchema()
  for path, v in list_tags(attrs):
    attr_name: str = path[-1]

    # handle date and number
    if attr_name in ['date', 'number', 'subject']:
      copy_leaf_tag(attr_name, v, tree)

    if attr_name.startswith('org-'):
      convert_org(attr_name, attr=v, dest=tree)

    if "sign_value_currency" == path[0]:
      convert_sign_value_currency(path, v, tree)

  return tree


def convert_sign_value_currency(path: [str], v, dest):
  if dest.price is None:
    dest.price = ContractPrice()

  if len(path) == 1:
    copy_attr(v, dest.price)

  elif len(path) == 2:
    handle_sign_value_currency(path[1:], v, dest.price)


def convert_protocol_db_attributes_to_tree(attrs) -> ProtocolSchema:
  tree = ProtocolSchema()

  for paths, v in list_tags(attrs):
    attr_name: str = paths[0]
    attr_name_clean, _i = index_of_key(attr_name)
    if ("agenda_item" == attr_name_clean):
      agenda_item_node = array_set_or_get_at(tree.agenda_items, _i, AgendaItem)
      if len(paths) == 1:
        copy_attr(v, dest=agenda_item_node)
      else:
        convert_agenda_item(paths[1:], v, agenda_item_node)
    # handle org
    elif attr_name.startswith('org-'):
      tree.org = map_org(attr_name, v, tree.org)

    # handle date and number
    elif (attr_name == 'date'):
      tree.date = map_tag(v)
    elif (attr_name == 'number'):
      tree.number = map_tag(v)

    elif (attr_name == 'org_structural_level'):
      tree.structural_level = map_tag(v)

  return tree


def convert_charter_db_attributes_to_tree(attrs):
  tree = CharterSchema()
  for key_s, v in list_tags(attrs):

    attr_name: str = key_s[-1]

    if attr_name in ['date', 'number']:
      copy_leaf_tag(attr_name, v, tree)

    # handle org
    elif attr_name.startswith('org-'):
      tree.org = map_org(attr_name, v, tree.org)

    # handle constraints
    elif (key_s[0] in OrgStructuralLevel._member_names_):
      structural_level_node = _find_by_value(tree.structural_levels, key_s[0])
      if structural_level_node is None:
        structural_level_node = CharterStructuralLevel()
        tree.structural_levels.append(structural_level_node)

      if len(key_s) == 1:
        copy_attr(v, dest=structural_level_node)
      else:
        convert_constraints(key_s[1:], v, structural_level_node)

  clean_up_tree(tree)
  return tree


def get_legacy_docs_ids() -> []:
  db = get_mongodb_connection()
  documents_collection = db['documents']

  _attr_updated_by_user = {
    '$and': [
      {'user.attributes_tree.creation_date': {'$exists': True}},
      {
        '$expr': {
          '$gt': ['$user.updateDate', '$user.attributes_tree.creation_date']
        }
      }
    ]
  }  # TODO: do something about this

  _small_version = {
    "user.attributes_tree.version.2": {"$lt": 7}
  }

  _no_user_tree = {"$and": [
    {"user.attributes": {'$exists': True}},
    {"user.attributes_tree": {'$exists': False}},
  ]}

  _no_attr_tree = {"$and": [
    {"analysis.attributes": {'$exists': True}},
    {"analysis.attributes_tree": {'$exists': False}},
  ]}

  query = {"$or": [
    _no_user_tree,
    _attr_updated_by_user,
    _no_attr_tree,
    _small_version]}

  cursor = documents_collection.find(query, projection={'_id': True})

  res = []
  for doc in cursor:
    res.append(doc["_id"])

  return res


def convert_one(db, doc: dict):
  migration_logger.info(f'updating {doc["_id"]} ....')

  kind: str = doc['parse']['documentType']
  kind = kind.lower()
  a_attr_tree = None
  u_attr_tree = None

  u = None
  a = doc['analysis'].get('attributes')
  if 'user' in doc:
    u = doc['user'].get('attributes')

  kind2method = {
    "protocol": convert_protocol_db_attributes_to_tree,
    "charter": convert_charter_db_attributes_to_tree,
    "contract": convert_contract_db_attributes_to_tree,
    'annex': convert_contract_db_attributes_to_tree,
    'supplementary_agreement': convert_contract_db_attributes_to_tree,
  }

  if kind in kind2method:
    a_attr_tree = {kind: kind2method[kind](a)}
    if u is not None:
      u_attr_tree = {kind: kind2method[kind](u)}

    a_attr_tree['version'] = analyser.__version_ints__
    a_attr_tree['creation_date'] = datetime.now()
    j, _ = to_json(a_attr_tree)
    db["documents"].update_one({'_id': doc["_id"]}, {"$set": {"analysis.attributes_tree": j}})
    migration_logger.debug(f'updated {kind} {doc["_id"]} analysis.attributes_tree')
    if u_attr_tree is not None:
      u_attr_tree['version'] = analyser.__version_ints__
      u_attr_tree['creation_date'] = datetime.now()
      j, _ = to_json(u_attr_tree)
      db["documents"].update_one({'_id': doc["_id"]}, {"$set": {"user.attributes_tree": j}})
      migration_logger.debug(f'updated {kind} {doc["_id"]} user.attributes_tree')


def should_i_migrate(ids) -> bool:
  if len(ids) == 0:
    migration_logger.info("Migration: no legacy docs found in DB")
    return False

  print(f">> {len(ids)} legacy doc(s) found in DB. ")

  if '-skipmigration' in sys.argv:
    print("migration skipped due to -skipmigration flag")
    return False

  if '-forcemigration' in sys.argv:
    return True

  print("use -forcemigration cmd line arg if your answer is always yes")
  print("use -skipmigration cmd line arg if your answer is always no")
  print('\a')
  print("If you want to convert (migrate) them, type YES (it's safe, trust me)")

  yesno = str(input())

  if yesno == 'YES':
    return True

  return False


def convert_all_docs():
  ids = get_legacy_docs_ids()
  if should_i_migrate(ids):

    db = get_mongodb_connection()
    documents_collection = db['documents']

    for id in ids:
      doc = documents_collection.find_one({"_id": id}, projection={
        '_id': True,
        'analysis.attributes': True,
        'user.attributes': True,
        'parse.documentType': True})

      convert_one(db, doc)

    migration_logger.info(f"converted {len(ids)} documents")
  else:
    print('Skipping migration. Re-run when you change your mind.')


# ---------------------------------------------------------- self-TESTS:

def _test_protocol():
  db = get_mongodb_connection()

  doc = get_doc_by_id(ObjectId('5df7a66b200a3f4d0fad786f'))  # protocol
  convert_one(db, doc)


def _test_charter():
  doc = get_doc_by_id(ObjectId('5f64161009d100a445b7b0d6'))
  a = doc['analysis']['attributes']
  tree = {"charter": convert_charter_db_attributes_to_tree(a)}

  j, json_str = to_json(tree)

  return j, json_str, doc


def _test_contract():
  doc = get_doc_by_id(ObjectId('5f0bb4bd138e9184feef1fa8'))
  a = doc['analysis']['attributes']
  tree = {"contract": convert_contract_db_attributes_to_tree(a)}

  j, json_str = to_json(tree)

  return j, json_str, doc


def _test_convert():
  # charter: 5f64161009d100a445b7b0d6
  # protocol: 5ded4e214ddc27bcf92dd6cc
  # contract: 5f0bb4bd138e9184feef1fa8

  db = get_mongodb_connection()
  _test_protocol()

  j, json_str, doc = _test_charter()
  validate(instance=json_str, schema=document_schemas, format_checker=FormatChecker())
  db["documents"].update_one({'_id': doc["_id"]}, {"$set": {"analysis.attributes_tree": j}})

  j, json_str, doc = _test_contract()
  validate(instance=json_str, schema=document_schemas, format_checker=FormatChecker())
  db["documents"].update_one({'_id': doc["_id"]}, {"$set": {"analysis.attributes_tree": j}})
