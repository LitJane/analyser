import json
import os

import gridfs
import pymongo
import requests
from bson import json_util, ObjectId
from jsonschema import ValidationError, FormatChecker, Draft7Validator

from analyser import finalizer
from analyser.charter_parser import CharterParser
from analyser.contract_parser import ContractParser, GenericParser
from analyser.finalizer import normalize_only_company_name, compare_ignore_case, check_compliance, save_violations
from analyser.legal_docs import LegalDocument
from analyser.log import logger
from analyser.parsing import AuditContext
from analyser.persistence import DbJsonDoc
from analyser.protocol_parser import ProtocolParser
from analyser.schemas import document_schemas
from analyser.structures import DocumentState
from gpn.gpn import subsidiaries
from integration import mail
from integration.classifier.search_text import wrapper, all_labels, label2id
from integration.classifier.sender import get_sender_judicial_org
from integration.db import get_mongodb_connection

schema_validator = Draft7Validator(document_schemas, format_checker=FormatChecker())

CHARTER = 'CHARTER'
CONTRACT = 'CONTRACT'
PROTOCOL = 'PROTOCOL'
classifier_url = os.environ.get('GPN_CLASSIFIER_SERVICE_URL')


class Runner:
  default_instance: 'Runner' = None

  def __init__(self, init_embedder=True):
    self.protocol_parser = ProtocolParser()
    self.contract_parser = ContractParser()
    self.charter_parser = CharterParser()
    self.generic_parser = GenericParser()

  def init_embedders(self):
    pass

  @staticmethod
  def get_instance(init_embedder=False) -> 'Runner':
    if Runner.default_instance is None:
      Runner.default_instance = Runner(init_embedder=init_embedder)
    return Runner.default_instance


class BaseProcessor:
  parser = None
  generic_parser = Runner.get_instance().generic_parser

  def preprocess(self, jdoc: DbJsonDoc, context: AuditContext) -> DbJsonDoc:
    # phase I
    # TODO: include phase I into phase II, remove phase I
    if jdoc.is_user_corrected():
      logger.info(f"skipping doc {jdoc.get_id()} because it is corrected by user")
      # TODO: update state?
    else:
      legal_doc = jdoc.asLegalDoc()
      self.generic_parser.find_attributes(legal_doc, context)

      if self.parser is not None:
        self.parser.find_org_date_number(legal_doc, context)

      jdoc = save_analysis(jdoc, legal_doc, state=DocumentState.Preprocessed.value)
      # print(jdoc.analysis)
    return jdoc

  def process(self, db_document: DbJsonDoc, audit, context: AuditContext) -> LegalDocument or None:
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
          self.generic_parser.find_attributes(legal_doc, context)
          if self.parser is not None:
            self.parser.find_attributes(legal_doc, context)
          save_analysis(db_document, legal_doc, state=DocumentState.Done.value)
          # ANALYSING

        logger.info(f'analysis saved, doc._id={legal_doc.get_id()}')
      else:
        logger.info(f"excluding doc {db_document.get_id()}")
        # we re not saving doc here cuz we had NOT search for attrs
        change_doc_state(db_document, state=DocumentState.Excluded.value)

    except Exception as err:
      logger.exception(f'cant process document {db_document.get_id()}')
      # TODO: do not save the entire doc here, data loss possible
      save_analysis(db_document, legal_doc, DocumentState.Error.value, db_document.retry_number + 1)

    return legal_doc

  def is_valid(self, audit, db_document: DbJsonDoc):
    import pytz
    utc = pytz.UTC

    # date must be ok
    # TODO: rename: -> is_eligible
    if audit.get('pre-check'):
      return True
    _date = db_document.get_date_value()
    if _date is not None:
      try:
        date_is_ok = utc.localize(audit["auditStart"]) <= _date.replace(tzinfo=utc) <= utc.localize(audit["auditEnd"])
      except TypeError as e:
        logger.exception(e)
        date_is_ok = False
    else:
      # if date not found, we keep processing the doc anyway
      date_is_ok = True

    # org filter must be ok
    _audit_subsidiary: str = audit["subsidiary"]["name"]
    org_is_ok = ("* Все ДО" == _audit_subsidiary) or (self._same_org(db_document, _audit_subsidiary))

    if not (org_is_ok and date_is_ok) and db_document.documentType == 'ANNEX':
      _document = finalizer.get_parent_doc(audit, db_document.get_id())
      jdoc = DbJsonDoc(_document)
      return self.is_valid(audit, jdoc)

    return org_is_ok and date_is_ok

  def _same_org(self, db_doc: DbJsonDoc, subsidiary: str) -> bool:
    org = finalizer.get_org(db_doc.get_attributes_tree())
    if org is not None and org.get('name') is not None:
      org_name = normalize_only_company_name(org['name'].get('value'))
      if compare_ignore_case(org_name, normalize_only_company_name(subsidiary)):
        return True
      for known_subsidiary in subsidiaries:
        if compare_ignore_case(known_subsidiary.get('_id'), subsidiary):
          for alias in known_subsidiary.get('aliases'):
            if compare_ignore_case(alias, subsidiary):
              return True
    return False


class GenericProcessor(BaseProcessor):
  def __init__(self):
    self.parser = None
    # Runner.get_instance().generic_parser


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
document_processors = {CONTRACT: contract_processor,
                       CHARTER: CharterProcessor(),
                       PROTOCOL: ProtocolProcessor(),
                       'ANNEX': contract_processor,
                       'SUPPLEMENTARY_AGREEMENT': contract_processor,
                       'AGREEMENT': contract_processor,
                       "GENERIC": GenericProcessor()}


def get_audits() -> [dict]:
  db = get_mongodb_connection()
  audits_collection = db['audits']

  cursor = audits_collection.find({'status': 'InWork'}).sort([("createDate", pymongo.ASCENDING)])
  res = []
  for audit in cursor:
    res.append(audit)
  return res


def get_audits_for_notification() -> [dict]:
  db = get_mongodb_connection()
  audits_collection = db['audits']

  cursor = audits_collection.find({'toBeApproved': True, 'additionalFields.external_source': 'email', 'additionalFields.compliance_protocol_praparation_email_sent':{'$ne': True}}).sort([("createDate", pymongo.ASCENDING)])
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


def save_analysis(db_document: DbJsonDoc, doc: LegalDocument, state: int, retry_number: int = 0) -> DbJsonDoc:
  # TODO: does not save attributes
  analyse_json_obj: dict = doc.to_json_obj()
  db = get_mongodb_connection()
  documents_collection = db['documents']
  db_document.analysis = analyse_json_obj
  db_document.state = state
  validate_json_schema(db_document)

  db_document.retry_number = retry_number
  documents_collection.update({'_id': doc.get_id()}, db_document.as_dict(), True)
  return db_document


def save_audit_practice(audit, classification_result, zip_classified):
  if audit['additionalFields']['external_source'] != 'email':
    zip_classified = False
  db = get_mongodb_connection()
  db['audits'].update_one({'_id': audit['_id']}, {
    "$set": {'classification_result': classification_result, "additionalFields.zip_classified": zip_classified}})


def save_errors(audit, errors):
  db = get_mongodb_connection()
  db["audits"].update_one({'_id': audit["_id"]}, {"$push": {"errors": {'$each': errors}}})


def change_doc_state(doc, state):
  db = get_mongodb_connection()
  db['documents'].update_one({'_id': doc.get_id()}, {"$set": {"state": state}})


def change_audit_status(audit, status):
  db = get_mongodb_connection()
  db["audits"].update_one({'_id': audit["_id"]}, {"$set": {"status": status}})


def is_well_parsed(document: DbJsonDoc):
  return document.parserResponseCode == 200


def need_analysis(document: DbJsonDoc) -> bool:
  _is_not_a_charter = document.documentType != "CHARTER"
  _well_parsed = is_well_parsed(document)

  _need_analysis = _well_parsed and (_is_not_a_charter or document.isActiveCharter())
  return _need_analysis


def get_doc4classification(audit):
  main_doc = None
  if audit.get('additionalFields', '').get('main_document_id') is not None:
    main_doc = finalizer.get_doc_by_id(audit['additionalFields']['main_document_id'])
    main_doc_type = main_doc['documentType']
    if main_doc['parserResponseCode'] == 200 and main_doc_type != 'SUPPLEMENTARY_AGREEMENT' and main_doc_type != 'ANNEX':
      return main_doc, True
  document_ids = get_docs_by_audit_id(audit["_id"], id_only=True, states=[0, 5, 10, 15])
  for document_id in document_ids:
    _document = finalizer.get_doc_by_id(document_id)
    if _document['parserResponseCode'] == 200:
      doc_type = _document['documentType']
      if doc_type == 'CONTRACT':
        return _document, False
  if main_doc is not None and main_doc['parserResponseCode'] == 200:
    return main_doc, True
  for document_id in document_ids:
    _document = finalizer.get_doc_by_id(document_id)
    if _document['parserResponseCode'] == 200:
      return _document, False


def get_doc_headline_safely(document):
  try:
    return document['paragraphs'][0]['paragraphHeader']['text']
  except:
    return None


def doc_classification(audit):
  try:
    logger.info(f'.....classifying audit {audit["_id"]}')
    doc4classification, main_doc = get_doc4classification(audit)
    classification_result = None
    errors = []
    if doc4classification['documentType'] in ['CONTRACT', 'AGREEMENT', 'SUPPLEMENTARY_AGREEMENT']:
      violations, errors = check_compliance(audit, doc4classification)
      compliance_mapping = next(filter(lambda x: x['_id'] == 1015, all_labels), None)
      if len(errors) > 0:
        mail.send_compliance_error_email(audit, errors, compliance_mapping['email'])
        if not audit.get('additionalFields', {}).get('compliance_info_email_sent', False):
          result = mail.send_compliance_info_email(audit)
          db = get_mongodb_connection()
          db["audits"].update_one({'_id': audit["_id"]}, {"$set": {"additionalFields.compliance_info_email_sent": result}})

      if len(violations) > 0:
        classification_result = [{'id': compliance_mapping['_id'], 'label': compliance_mapping['label'], 'score': 1.0}]

    # detecting judicial organisation in sender (email_from) field
    doc_headline = get_doc_headline_safely(doc4classification['parse'])

    sender_judicial_org = None
    if audit['additionalFields']['external_source'] == 'email':
      sender_ = audit['additionalFields']['email_from']
      sender_judicial_org = get_sender_judicial_org(sender_)

    if (doc_headline is not None) and (sender_judicial_org is None):
      sender_judicial_org = get_sender_judicial_org(doc_headline)

    if sender_judicial_org is not None:
      classification_result = apply_judical_practice(classification_result, sender_judicial_org)

    if classification_result is None and len(errors) == 0:
      if classifier_url is None:
        classification_result = wrapper(doc4classification['parse'])
      else:
        response = requests.post(classifier_url + '/api/classify', json=doc4classification['parse'])
        if response.status_code != 200:
          logger.error(f'Classifier returned error code: {response.status_code}, message: {response.json()}')
          audits = get_mongodb_connection()['audits']
          update = {'$push': {'errors': {'type': 'classifier_service', 'text': 'Ошибка классификатора'}}}
          audits.update_one({'_id': ObjectId(audit["_id"])}, update)
          return
        classification_result = response.json()

    if classification_result:
        save_audit_practice(audit, classification_result, not main_doc)
        if audit['additionalFields']['external_source'] == 'email':
          top_result = next(filter(lambda x: x['_id'] == classification_result[0]['id'], all_labels), None)
          attachments = []
          fs = gridfs.GridFS(get_mongodb_connection())
          for file_id in audit['additionalFields']['file_ids']:
            attachments.append(fs.get(file_id))
          mail.send_classifier_email(audit, top_result, attachments, all_labels)
  except Exception as ex:
    logger.exception(ex)


def apply_judical_practice(classification_result, sender_judicial_org):
  if sender_judicial_org is not None:

    _l = 'Практика судебной защиты'
    _result = {
      'id': label2id[_l],
      'label': _l,
      'score': 1,
      'sender_judicial_org': sender_judicial_org
    }

    if not classification_result:
      classification_result = []

    classification_result.insert(0, _result)
  return classification_result


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
    audit_phase_1_doc(document_id, ctx, k, len(document_ids))


def audit_phase_1_doc(document_id, ctx, _k=1, _total=1):
  _document = finalizer.get_doc_by_id(document_id)
  jdoc = DbJsonDoc(_document)

  logger.info(f'......pre-pre-processing {_k} of {_total}  {jdoc.documentType}:{document_id}')

  processor: BaseProcessor = document_processors.get(jdoc.documentType)
  if processor is None:
    logger.warning(f'unknown/unsupported doc type: {jdoc.documentType},  using just generic processor {document_id}')
    if is_well_parsed(jdoc):
      ##finding common things, like case numbers, etc...
      document_processors.get('GENERIC').preprocess(jdoc=jdoc, context=ctx)
  else:
    logger.info(f'......pre-processing {_k} of {_total}  {jdoc.documentType}:{document_id}')
    if need_analysis(jdoc) and jdoc.isNew():
      processor.preprocess(jdoc=jdoc, context=ctx)


def audit_phase_2(audit, kind=None):
  # if audit.get('pre-check') and audit.get('checkTypes') is not None and len(audit['checkTypes']) == 0:
  #   change_audit_status(audit, "Finalizing")
  #   return

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

  if audit.get('pre-check') and 'Classification' in audit.get('checkTypes', {}):
    doc_classification(audit)
  if audit.get('pre-check') and 'Compliance' in audit.get('checkTypes', {}):
    document, _ = get_doc4classification(audit)
    violations, errors = check_compliance(audit, document)
    save_errors(audit, errors)
    save_violations(audit, violations)

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

  for audit in get_audits():
    # -----------------------
    # I
    logger.info('-> PHASE I...')
    audit_phase_1(audit, kind)

    # -----------------------
    # II
    if run_pahse_2:
      logger.info('-> PHASE II..')
      # phase 2
      audit_phase_2(audit, kind)
    else:
      logger.info("phase 2 is skipped")

  # -----------------------
  # III

  logger.info('-> PHASE III (finalize)...')
  finalizer.finalize()

  logger.info('-> PHASE IV (notifications)...')
  db = get_mongodb_connection()
  for audit in get_audits_for_notification():
    result = mail.send_compliance_protocol_preparation_email(audit)
    db["audits"].update_one({'_id': audit["_id"]}, {"$set": {"additionalFields.compliance_protocol_praparation_email_sent": result}})


if __name__ == '__main__':
  run()
