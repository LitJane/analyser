import datetime
import re

from analyser.hyperparams import HyperParameters
from analyser.legal_docs import LegalDocument
from analyser.ml_tools import SemanticTag

'''
refer https://github.com/nemoware/document-parser/blob/24013f562a8bc853134e116531f06ab9edcc0b00/src/main/java/com/nemo/document/parser/DocumentParser.java#L29
'''

months_short_temp = [r"янв", r"фев", r"мар", r"апр", r"ма[йя]", r"июн",
                     r"июл", r"авг", r"сен", r"окт", r"ноя", r"дек"]
months_short = [re.compile(c, re.UNICODE | re.IGNORECASE) for c in months_short_temp]
_months_short_combined = '|'.join(months_short_temp)
_month_no = r'1[0-2]|0[1-9]'
_months_long_combined = 'января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря'
_date_month_ = f'({_months_short_combined})|({_months_long_combined})|({_month_no})'

_date_day = r'«?(?P<day>[1-2][0-9]|3[01]|0?[1-9])»?'
_date_year = r'(?P<year>[1-2]\d{3})'
_date_month = f'(?P<month>{_date_month_})'
_date_separator = r'(\s*|\-|\.)'
date_regex_str = f'{_date_day}{_date_separator}{_date_month}{_date_separator}{_date_year}'
date_regex_c = re.compile(date_regex_str, re.IGNORECASE | re.UNICODE)


def find_document_date(ldoc: LegalDocument, tagname='date') -> SemanticTag or None:
  head: LegalDocument = get_doc_head(ldoc)
  c_span, _date = find_date(head.text)
  if c_span is None:
    return None
  span = head.tokens_map.token_indices_by_char_range(c_span)
  return SemanticTag(tagname, _date, span)


def find_date(text: str) -> ([], datetime.datetime):
  try:
    findings = re.finditer(date_regex_c, text)
    if findings:
      finding = next(findings)
      _date = parse_date(finding)
      if _date:
        return finding.span(), _date
  except Exception:
    pass

  return None, None


def get_doc_head(ldoc: LegalDocument) -> LegalDocument:
  if ldoc.paragraphs:
    headtag: SemanticTag = ldoc.paragraphs[0].as_combination()
    if len(headtag) > 50:
      return ldoc[headtag.as_slice()]
  # fallback
  return ldoc[0:HyperParameters.protocol_caption_max_size_words]


def parse_date(finding) -> ([], datetime.datetime):
  month = _get_month_number(finding["month"])
  year = int(finding['year'])
  day = int(finding['day'])

  if month > 0:
    _date = datetime.datetime(year, month, day)
    return _date


def _get_month_number(m):
  if m.isdigit():
    try:
      return int(m)
    except Exception:
      pass

  for p, month in enumerate(months_short):
    if re.match(month, m):
      return p + 1
  return -1
