import re

from analyser.doc_dates import get_doc_head
from analyser.legal_docs import LegalDocument
from analyser.ml_tools import SemanticTag

doc_num_exclude_prefixes = r"(?<!лицензии )(?<!лицензнии )(?<!доверенности )"  # возможно надо исключить номера соглашений, тогда добавить здесь (?<!соглашения )
document_number_c = re.compile(
  doc_num_exclude_prefixes + r"(договор(?=\s\d)|№|N|#)\s*(?P<number>(?!на|от)([a-zа-я0-9_]+((/|-|\s(?=\d))[a-zа-я0-9_]+)*))",
  re.IGNORECASE)
document_number_valid_c = re.compile(r"([A-ZА-Я0-9]+)")


def is_number_valid(_number):
  return document_number_valid_c.match(_number)


def find_document_number_span(head_text: str) -> SemanticTag or None:
  try:
    findings = re.finditer(document_number_c, head_text)
    if findings:
      finding = next(findings)
      _number = finding['number']
      if is_number_valid(_number):
        return _number, finding.span()
  except Exception:
    pass
  return None, None


def find_document_number(doc: LegalDocument, tagname='number') -> SemanticTag or None:
  head: LegalDocument = get_doc_head(doc)

  _number, finding_span = find_document_number_span(head.text)
  if _number is not None:
    span = head.tokens_map.token_indices_by_char_range(finding_span)
    return SemanticTag(tagname, _number, span)

  return None


def find_document_number_in_subdoc(doc: LegalDocument, tagname='number', parent=None) -> [SemanticTag]:
  ret = []
  findings = re.finditer(document_number_c, doc.text)
  if findings:
    for finding in findings:
      _number = finding['number']
      if is_number_valid(_number):
        span = doc.tokens_map.token_indices_by_char_range(finding.span())
        tag = SemanticTag(tagname, _number, span, parent=parent)
        tag.offset(doc.start)
        ret.append(tag)
      else:
        print('invalid', _number)

  return ret
