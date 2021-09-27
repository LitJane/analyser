import json
import traceback

import pymongo
from bson import json_util
from jsonschema import ValidationError, FormatChecker, Draft7Validator

from analyser import finalizer
from analyser.charter_parser import CharterParser
from analyser.contract_parser import ContractParser
from analyser.finalizer import normalize_only_company_name, compare_ignore_case
from analyser.legal_docs import LegalDocument
from analyser.log import logger
from analyser.parsing import AuditContext
from analyser.persistence import DbJsonDoc
from analyser.protocol_parser import ProtocolParser
from analyser.schemas import document_schemas
from analyser.structures import DocumentState
from integration.db import get_mongodb_connection

schema_validator = Draft7Validator(document_schemas, format_checker=FormatChecker())

CHARTER = 'CHARTER'
CONTRACT = 'CONTRACT'
PROTOCOL = 'PROTOCOL'


class Runner:
  default_instance: 'Runner' = None

  def __init__(self, init_embedder=True):
    self.protocol_parser = ProtocolParser()
    self.contract_parser = ContractParser()
    self.charter_parser = CharterParser()

  def init_embedders(self):
    pass

  @staticmethod
  def get_instance(init_embedder=False) -> 'Runner':
    if Runner.default_instance is None:
      Runner.default_instance = Runner(init_embedder=init_embedder)
    return Runner.default_instance


class BaseProcessor:
  parser = None

  def preprocess(self, jdoc: DbJsonDoc, context: AuditContext):
    # phase I
    # TODO: include phase I into phase II, remove phase I
    if jdoc.is_user_corrected():
      logger.info(f"skipping doc {jdoc.get_id()} because it is corrected by user")
      # TODO: update state?
    else:
      legal_doc = jdoc.asLegalDoc()
      self.parser.find_org_date_number(legal_doc, context)
      save_analysis(jdoc, legal_doc, state=DocumentState.Preprocessed.value)
      return legal_doc

  def process(self, db_document: DbJsonDoc, audit, context: AuditContext) -> LegalDocument:
    # phase II
    if db_document.retry_number is None:
      db_document.retry_number = 0

    if db_document.retry_number > 2:
      logger.info(
        f'{db_document.documentType} {db_document.get_id()} exceeds maximum retries for analysis and is skipped')
      return None

    legal_doc = db_document.asLegalDoc()
    try:

      # self.parser.find_org_date_number(legal_doc, context) # todo: remove this call
      # todo: make sure it is done in phase 1, BUT phase 1 is deprecated ;-)
      # save_analysis(db_document, legal_doc, state=DocumentState.InWork.value)

      if audit is None or self.is_valid(audit, db_document):

        if db_document.is_user_corrected():
          logger.info(f"skipping doc {db_document.get_id()} postprocessing because it is corrected by user")
          change_doc_state(db_document, state=DocumentState.Done.value)
        else:
          # ANALYSING
          self.parser.find_attributes(legal_doc, context)
          save_analysis(db_document, legal_doc, state=DocumentState.Done.value)
          # ANALYSING

        logger.info(f'analysis saved, doc._id={legal_doc.get_id()}')
      else:
        logger.info(f"excluding doc {db_document.get_id()}")
        # we re not saving doc here cuz we had NOT search for attrs
        change_doc_state(db_document, state=DocumentState.Excluded.value)

    except Exception as err:
      traceback.print_tb(err.__traceback__)
      logger.exception(f'cant process document {db_document.get_id()}')
      # TODO: do not save the entire doc here, data loss possible
      save_analysis(db_document, legal_doc, DocumentState.Error.value, db_document.retry_number + 1)

    return legal_doc

  def is_valid(self, audit, db_document: DbJsonDoc):
    # date must be ok
    # TODO: rename: -> is_eligible
    if audit.get('pre-check'):
      return True
    _date = db_document.get_date_value()
    if _date is not None:
      date_is_ok = audit["auditStart"] <= _date <= audit["auditEnd"]
    else:
      # if date not found, we keep processing the doc anyway
      date_is_ok = True

    # org filter must be ok
    _audit_subsidiary: str = audit["subsidiary"]["name"]
    org_is_ok = ("* Все ДО" == _audit_subsidiary) or (self._same_org(db_document, _audit_subsidiary))

    return org_is_ok and date_is_ok

  def _same_org(self, db_doc: DbJsonDoc, subsidiary: str) -> bool:
    org = finalizer.get_org(db_doc.get_attributes_tree())
    if org is not None and org.get('name') is not None:
      org_name = normalize_only_company_name(org['name'].get('value').strip().replace('"', '').replace("'", '').replace('«', '').replace('»', ''))
      if compare_ignore_case(org_name, subsidiary):
        return True
    return False


class ProtocolProcessor(BaseProcessor):
  def __init__(self):
    self.parser = Runner.get_instance().protocol_parser


class CharterProcessor(BaseProcessor):
  def __init__(self):
    self.parser = Runner.get_instance().charter_parser


class ContractProcessor(BaseProcessor):
  def __init__(self):
    self.parser = Runner.get_instance().contract_parser


contract_processor = ContractProcessor()
document_processors = {CONTRACT: contract_processor, CHARTER: CharterProcessor(), "PROTOCOL": ProtocolProcessor(),
                       'ANNEX': contract_processor, 'SUPPLEMENTARY_AGREEMENT': contract_processor, 'AGREEMENT': contract_processor}


def get_audits() -> [dict]:
  db = get_mongodb_connection()
  audits_collection = db['audits']

  cursor = audits_collection.find({'status': 'InWork'}).sort([("createDate", pymongo.ASCENDING)])
  res = []
  for audit in cursor:
    res.append(audit)
  return res


def get_all_new_charters():
  # TODO: fetch chartes with unknown satate (might be)
  return get_docs_by_audit_id(id=None, states=[DocumentState.New.value], kind=CHARTER)


def get_docs_by_audit_id(id: str or None, states=None, kind=None, id_only=False) -> []:
  db = get_mongodb_connection()
  documents_collection = db['documents']

  query = {
    "$and": [
      {'auditId': id},
      {"parserResponseCode": 200},
      {"$or": [{"analysis.version": None},
               # {"analysis.version": {"$ne": analyser.__version__}},
               {"state": None}]}
    ]
  }

  if states is not None:
    for state in states:
      query["$and"][2]["$or"].append({"state": state})

  if kind is not None:
    query["$and"].append({'documentType': kind})

  if id_only:
    cursor = documents_collection.find(query, projection={'_id': True})
  else:
    cursor = documents_collection.find(query)

  res = []
  for doc in cursor:
    if id_only:
      res.append(doc["_id"])
    else:
      res.append(doc)
  return res


def validate_json_schema(db_document):
  try:
    json_str = json.dumps(db_document.analysis['attributes_tree'], ensure_ascii=False,
                          default=json_util.default)
    schema_validator.validate(json_str)
  except ValidationError as e:
    logger.error(e)
    db_document.state = DocumentState.Error.value


def save_analysis(db_document: DbJsonDoc, doc: LegalDocument, state: int, retry_number: int = 0):
  # TODO: does not save attributes
  analyse_json_obj: dict = doc.to_json_obj()
  db = get_mongodb_connection()
  documents_collection = db['documents']
  db_document.analysis = analyse_json_obj
  db_document.state = state
  validate_json_schema(db_document)

  db_document.retry_number = retry_number
  documents_collection.update({'_id': doc.get_id()}, db_document.as_dict(), True)


def change_doc_state(doc, state):
  db = get_mongodb_connection()
  db['documents'].update_one({'_id': doc.get_id()}, {"$set": {"state": state}})


def change_audit_status(audit, status):
  db = get_mongodb_connection()
  db["audits"].update_one({'_id': audit["_id"]}, {"$set": {"status": status}})


def need_analysis(document: DbJsonDoc) -> bool:
  _is_not_a_charter = document.documentType != "CHARTER"
  _well_parsed = document.parserResponseCode == 200

  _need_analysis = _well_parsed and (_is_not_a_charter or document.isActiveCharter())
  return _need_analysis


def audit_phase_1(audit, kind=None):
  logger.info(f'.....processing audit {audit["_id"]}')
  if audit.get('subsidiary') is None:
    ctx = AuditContext()
  else:
    ctx = AuditContext(audit["subsidiary"]["name"])

  document_ids = get_docs_by_audit_id(audit["_id"], states=[DocumentState.New.value], kind=kind, id_only=True)
  _charter_ids = audit.get("charters", [])
  document_ids.extend(_charter_ids)

  for k, document_id in enumerate(document_ids):
    _document = finalizer.get_doc_by_id(document_id)
    jdoc = DbJsonDoc(_document)

    processor: BaseProcessor = document_processors.get(jdoc.documentType)
    if processor is None:
      logger.warning(f'unknown/unsupported doc type: {jdoc.documentType}, cannot process {document_id}')
    else:
      logger.info(f'......pre-processing {k} of {len(document_ids)}  {jdoc.documentType}:{document_id}')
      if need_analysis(jdoc) and jdoc.isNew():
        processor.preprocess(jdoc=jdoc, context=ctx)


def audit_phase_2(audit, kind=None):
  if audit.get('subsidiary') is None:
    ctx = AuditContext()
  else:
    ctx = AuditContext(audit["subsidiary"]["name"])

  print(f'.....processing audit {audit["_id"]}')

  document_ids = get_docs_by_audit_id(audit["_id"],
                                      states=[DocumentState.Preprocessed.value, DocumentState.Error.value],
                                      kind=kind, id_only=True)

  _charter_ids = audit.get("charters", [])
  document_ids.extend(_charter_ids)

  for k, document_id in enumerate(document_ids):
    _document = finalizer.get_doc_by_id(document_id)
    jdoc = DbJsonDoc(_document)

    processor: BaseProcessor = document_processors.get(jdoc.documentType)
    if processor is None:
      logger.warning(f'unknown/unsupported doc type: {jdoc.documentType}, cannot process {document_id}')
    else:
      if need_analysis(jdoc) and jdoc.isPreprocessed():
        logger.info(f'.....processing  {k} of {len(document_ids)}   {jdoc.documentType} {document_id}')
        processor.process(jdoc, audit, ctx)

  change_audit_status(audit, "Finalizing")  # TODO: check ALL docs in proper state


def audit_charters_phase_1():
  """preprocess"""
  charters = get_all_new_charters()
  processor: BaseProcessor = document_processors[CHARTER]

  for k, charter in enumerate(charters):
    jdoc = DbJsonDoc(charter)
    logger.info(f'......pre-processing {k} of {len(charters)} CHARTER {jdoc.get_id()}')
    ctx = AuditContext()
    processor.preprocess(jdoc, context=ctx)


def audit_charters_phase_2():  # XXX: #TODO: DO NOT LOAD ALL CHARTERS AT ONCE
  charters = get_docs_by_audit_id(id=None, states=[DocumentState.Preprocessed.value, DocumentState.Error.value],
                                  kind=CHARTER)

  for k, _document in enumerate(charters):
    jdoc = DbJsonDoc(_document)
    processor: BaseProcessor = document_processors[CHARTER]

    logger.info(f'......processing  {k} of {len(charters)}  CHARTER {jdoc.get_id()}')
    ctx = AuditContext()
    processor.process(jdoc, audit=None, context=ctx)


def run(run_pahse_2=True, kind=None):
  # -----------------------
  logger.info('-> PHASE 0 (charters)...')
  # NIL (сорян, в системе римских цифр отсутствует ноль)
  audit_charters_phase_1()
  if run_pahse_2:
    audit_charters_phase_2()

  # -----------------------
  # I
  logger.info('-> PHASE I...')
  for audit in get_audits():
    audit_phase_1(audit, kind)

  # -----------------------
  # II
  logger.info('-> PHASE II..')
  if run_pahse_2:
    # phase 2
    for audit in get_audits():
      audit_phase_2(audit, kind)

  else:
    logger.info("phase 2 is skipped")

  # -----------------------
  # III
  logger.info('-> PHASE III (finalize)...')
  finalizer.finalize()


if __name__ == '__main__':
  run()
