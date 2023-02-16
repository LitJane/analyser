from bson import ObjectId

from analyser.finalizer import get_doc_by_id, get_audit_by_id
from analyser.log import logger
from analyser.parsing import AuditContext
from analyser.persistence import DbJsonDoc
from analyser.runner import BaseProcessor, document_processors, CONTRACT


def re_analyze_contract(self):
  processor: BaseProcessor = document_processors[CONTRACT]
  # doc = get_doc_by_id(ObjectId('638f0a81b1363747e929f304'))
  doc = get_doc_by_id(ObjectId('63c506d4e2456d59975e1386'))
  #
  if doc is None:
    raise RuntimeError("fix unit test please, doc with given UID is not in test DB")

  audit = get_audit_by_id(doc['auditId'])
  # print(audit)

  jdoc = DbJsonDoc(doc)
  logger.info(f'......pre-processing {jdoc._id}')
  ctx = AuditContext()
  ctx.audit_subsidiary_name = audit.get('subsidiary', {}).get('name')
  processor.preprocess(jdoc, context=ctx)
  processor.process(jdoc, audit, ctx)