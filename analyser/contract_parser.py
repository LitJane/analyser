from enum import Enum

import numpy as np
import pandas as pd
from overrides import overrides
from pandas import DataFrame

from analyser.attributes import to_json
from analyser.contract_agents import ContractAgent, normalize_contract_agent
from analyser.doc_dates import find_date
from analyser.documents import TextMap
from analyser.hyperparams import HyperParameters
from analyser.insides_finder import InsidesFinder
from analyser.legal_docs import LegalDocument, ContractValue, ParserWarnings, find_value_sign
from analyser.log import logger
from analyser.ml_tools import SemanticTag, SemanticTagBase, is_span_intersect
from analyser.parsing import ParsingContext, AuditContext, find_value_sign_currency_attention
from analyser.patterns import AV_SOFT, AV_PREFIX
from analyser.schemas import ContractSchema, OrgItem, ContractPrice, merge_spans
from analyser.text_normalize import r_human_name_compilled
from analyser.text_tools import to_float
from analyser.transaction_values import ValueSpansFinder
from tf_support.tf_subject_model import load_subject_detection_trained_model, decode_subj_prediction, \
  nn_predict


class ContractDocument(LegalDocument):

  def __init__(self, original_text):
    LegalDocument.__init__(self, original_text)

    self.attributes_tree = ContractSchema()

  def to_json_obj(self) -> dict:
    j: dict = super().to_json_obj()
    _attributes_tree_dict, _ = to_json(self.attributes_tree)
    j['attributes_tree'] = {"contract": _attributes_tree_dict}
    return j

  def get_number(self) -> SemanticTagBase:
    return self.attributes_tree.number

  def set_number(self, number: SemanticTagBase):
    self.attributes_tree.number = number

  def get_date(self) -> SemanticTagBase:
    return self.attributes_tree.date

  def set_date(self, date: SemanticTagBase):
    self.attributes_tree.date = date

  def get_subject(self) -> SemanticTagBase:
    return self.attributes_tree.subject

  def set_subject(self, subject: SemanticTagBase):
    self.attributes_tree.subject = subject

  subject = property(get_subject, set_subject)
  date = property(get_date, set_date)
  number = property(get_number, set_number)


ContractDocument3 = ContractDocument


class ContractParser(ParsingContext):

  def __init__(self, embedder=None, sentence_embedder=None):
    ParsingContext.__init__(self, embedder, sentence_embedder)
    self.subject_prediction_model = load_subject_detection_trained_model()
    self.insides_finder = InsidesFinder()

  def find_org_date_number(self, contract: ContractDocument, ctx: AuditContext) -> ContractDocument:

    _head = contract[0:300]  # warning, trimming doc for analysis phase 1
    if _head.embeddings is None:
      logger.debug('embedding 300-trimmed contract')
      _head.embedd_tokens(self.get_embedder())

    # predicting with NN
    logger.debug('predicting semantic_map in 300-trimmed contract with NN')
    semantic_map, _ = nn_predict(self.subject_prediction_model, _head)

    contract.attributes_tree.orgs = nn_find_org_names(_head.tokens_map, semantic_map,
                                                      audit_ctx=ctx)
    check_orgs_natural_person(contract.attributes_tree.orgs, contract.get_headline())  # mutator

    # TODO: maybe move contract.tokens_map into text map
    contract.attributes_tree.number = nn_get_contract_number(_head.tokens_map, semantic_map)
    contract.attributes_tree.date = nn_get_contract_date(_head.tokens_map, semantic_map)

    return contract

  def validate(self, document: ContractDocument, ctx: AuditContext):
    document.clear_warnings()

    if not document.attributes_tree.orgs:
      document.warn(ParserWarnings.org_name_not_found)

    if not document.date:
      document.warn(ParserWarnings.date_not_found)

    if not document.number:
      document.warn(ParserWarnings.number_not_found)

    if not document.attributes_tree.price:
      document.warn(ParserWarnings.contract_value_not_found)

    if not document.subject:
      document.warn(ParserWarnings.contract_subject_not_found)

    self.log_warnings()

  @overrides
  def find_attributes(self, contract: ContractDocument, ctx: AuditContext) -> ContractDocument:
    """
    this analyser should care about embedding, because it decides wheater it needs (NN) embeddings or not
    """
    self._reset_context()
    contract = self.find_org_date_number(contract, ctx)

    _contract_cut = contract
    if len(contract) > HyperParameters.max_doc_size_tokens:
      contract.warn_trimmed(HyperParameters.max_doc_size_tokens)
      _contract_cut = contract[
                      0:HyperParameters.max_doc_size_tokens]  # warning, trimming doc for analysis phase 1

    # ------ lazy embedding
    if _contract_cut.embeddings is None:
      _contract_cut.embedd_tokens(self.get_embedder())

    # -------------------------------
    # repeat phase 1

    # self.find_org_date_number(contract, ctx)
    semantic_map, subj_1hot = nn_predict(self.subject_prediction_model, _contract_cut)

    if not contract.attributes_tree.number:
      contract.attributes_tree.number = nn_get_contract_number(_contract_cut.tokens_map, semantic_map)

    if not contract.date:
      contract.date = nn_get_contract_date(_contract_cut.tokens_map, semantic_map)
    #
    # # -------------------------------
    # # -------------------------------orgs, agents
    if (contract.attributes_tree.orgs is None) or len(contract.attributes_tree.orgs) < 2:
      contract.attributes_tree.orgs = nn_find_org_names(_contract_cut.tokens_map, semantic_map,
                                                        audit_ctx=ctx)

    check_orgs_natural_person(contract.attributes_tree.orgs, contract.get_headline())  # mutator

    # -------------------------------subject
    contract.subject = nn_get_subject(_contract_cut.tokens_map, semantic_map, subj_1hot)

    # -------------------------------values, prices, amounts
    self._logstep("finding contract values")
    contract.contract_values = nn_find_contract_value(_contract_cut.tokens_map, semantic_map)

    if len(contract.contract_values) > 0:
      contract.attributes_tree.price = contract.contract_values[0]
    # TODO: convert price!!

    # --------------------------------------insider
    self._logstep("finding insider info")
    self.insides_finder.find_insides(contract)

    # --------------------------------------
    self.validate(contract, ctx)
    return contract


ContractAnlysingContext = ContractParser  ##just alias, for ipnb compatibility. TODO: remove


def max_confident(vals: [ContractPrice]) -> ContractPrice or None:
  if len(vals) == 0:
    return None
  return max(vals, key=lambda a: a.integral_sorting_confidence())


def max_value(vals: [ContractValue]) -> ContractValue or None:
  if len(vals) == 0:
    return None
  return max(vals, key=lambda a: a.value.value)


def _sub_attention_names(subj: Enum):
  a = f'x_{subj}'
  b = AV_PREFIX + f'x_{subj}'
  c = AV_SOFT + a
  return a, b, c


def nn_find_org_names(textmap: TextMap, semantic_map: DataFrame,
                      audit_ctx: AuditContext) -> [ContractAgent]:
  contract_agents: [ContractAgent] = []

  types = nn_get_tag_values('org-type', textmap, semantic_map, max_tokens=12, threshold=0.5,
                            limit=2)
  names = nn_get_tag_values('org-name', textmap, semantic_map, max_tokens=12, threshold=0.5,
                            limit=2)
  aliases = nn_get_tag_values('org-alias', textmap, semantic_map, max_tokens=4, threshold=0.5,
                              limit=2)

  for o in [0, 1]:
    ca: ContractAgent = ContractAgent()
    if len(names) > o:
      ca.name = names[o]
    if len(aliases) > o:
      ca.alias = aliases[o]
    if len(types) > o:
      ca.type = types[o]
    normalize_contract_agent(ca)
    contract_agents.append(ca)

  contract_agents = [c for c in contract_agents if c.is_valid()]

  def _name_val_safe(a):
    if a.name is not None:
      return a.name.value
    return ''

  if audit_ctx.audit_subsidiary_name:
    # known subsidiary goes first
    contract_agents = sorted(contract_agents, key=lambda a: not audit_ctx.is_same_org(_name_val_safe(a)))
  else:
    contract_agents = sorted(contract_agents, key=lambda a: _name_val_safe(a))
    contract_agents = sorted(contract_agents, key=lambda a: not a.is_known_subsidiary)

  contract_agents = check_org_intersections(contract_agents)  # mutator

  return contract_agents  # _swap_org_tags(cas)


def check_orgs_natural_person(contract_agents: [OrgItem], header0: str):
  if not contract_agents:
    return

  for contract_agent in contract_agents:
    check_org_is_natural_person(contract_agent)

  logger.info(f'header: {header0}')

  if header0:
    if header0.lower().find('с физическим лицом') >= 0:
      _set_natural_person(contract_agents[-1])

  return contract_agents


def check_org_is_natural_person(contract_agent: OrgItem):
  human_name = False
  if contract_agent.name is not None:
    name = contract_agent.name.value
    x = r_human_name_compilled.search(name)
    if x is not None:
      human_name = True

  if human_name:
    _set_natural_person(contract_agent)


def _set_natural_person(contract_agent: OrgItem):
  if contract_agent.type is None:
    contract_agent.type = SemanticTag(None, None)

  contract_agent.type.value = 'Физическое лицо'


def check_org_intersections(contract_agents: [OrgItem]):
  '''
  achtung, darling! das ist mutator metoden, ja
  :param contract_agents:
  :return:
  '''
  if len(contract_agents) < 2:
    return
  if contract_agents[0].alias is None:
    return
  if contract_agents[1].alias is None:
    return

  crossing = is_span_intersect(
    contract_agents[0].alias.span,
    contract_agents[1].alias.span
  )
  if crossing:
    # keep most confident, remove another
    if contract_agents[0].alias.confidence < contract_agents[1].alias.confidence:
      contract_agents[0].alias = None  # Sorry =(
    else:
      contract_agents[1].alias = None  # Sorry =( You must not conflict

  return contract_agents


def nn_find_contract_value(textmap:TextMap, tagsmap: DataFrame) -> [ContractPrice]:
  cp = ContractPrice()

  parent_tag = nn_get_tag_values('value', textmap, tagsmap, max_tokens=50, threshold=0.02, limit=1, return_single=True)

  cp.amount = nn_get_tag_values('amount', textmap, tagsmap, max_tokens=4, threshold=0.4, limit=1, return_single=True)
  cp.sign = nn_get_tag_values('sign', textmap, tagsmap, max_tokens=10, threshold=0.03, limit=1, return_single=True)
  cp.currency = nn_get_tag_values('currency', textmap, tagsmap, max_tokens=4, threshold=0.3, limit=1,
                                  return_single=True)
  cp.amount_brutto = nn_get_tag_values('amount_brutto', textmap, tagsmap, max_tokens=4, threshold=0.4, limit=1,
                                       return_single=True)
  cp.amount_netto = nn_get_tag_values('amount_netto', textmap, tagsmap, max_tokens=4, threshold=0.04, limit=1,
                                      return_single=True)
  cp.vat = nn_get_tag_values('vat', textmap, tagsmap, max_tokens=4, threshold=0.02, limit=1, return_single=True)

  all_none = True
  for p in cp.list_children():
    if p is not None:
      all_none = False
  if all_none:
    return []

  #   ///SIGN

  if cp.sign:
    _start = cp.sign.span[0] - 5
    region = textmap.slice(slice(_start, cp.sign.span[0] + 5))
    _sign, _sign_span = find_value_sign(region)
    cp.sign.value = _sign
    if _sign_span is not None:
      cp.sign.span = _sign_span
      cp.sign.offset(_start)

  cp.span = merge_spans(cp.list_children())
  if parent_tag:
    cp.span = merge_spans([cp, parent_tag])

  try:
    region = textmap.text_range(cp.span)
    results = ValueSpansFinder(region)

    if cp.currency is None:
      cp.currency = SemanticTag('currency')
    cp.currency.span = textmap.token_indices_by_char_range(results.currency_span)
    cp.currency.value = results.currencly_name
    cp.currency.offset(cp.span[0])

  except TypeError as e:
    logger.error(e)
    results = None

  try:
    cp.amount.value = to_float(cp.amount.value)
  except Exception as e:
    logger.error(e)

  try:
    cp.vat.value = to_float(cp.vat.value)
  except Exception as e:
    logger.error(e)

  try:
    cp.amount_netto.value = to_float(cp.amount_netto.value)
  except Exception as e:
    logger.error(e)

  try:
    cp.amount_brutto.value = to_float(cp.amount_brutto.value)
  except Exception as e:
    logger.error(e)

  if (results is not None):
    if results.including_vat:
      cp.amount_brutto = cp.amount
      cp.amount_netto = None
    else:
      cp.amount_netto = cp.amount
      cp.amount_brutto = None

  return [cp]


def nn_get_subject(textmap: TextMap, semantic_map: DataFrame, subj_1hot) -> SemanticTag:
  predicted_subj_name, confidence, _ = decode_subj_prediction(subj_1hot)

  tags = nn_get_tag_values('subject', textmap, semantic_map, max_tokens=200, threshold=0.02,
                           limit=1)

  span = None
  if tags:
    tag = tags[0]
    span = tag.span
  tag = SemanticTag(None, predicted_subj_name.name, span=span, confidence=confidence)

  return tag


def nn_get_contract_number(textmap: TextMap, semantic_map: DataFrame) -> SemanticTag:
  tags = nn_get_tag_values('number', textmap, semantic_map, max_tokens=5, threshold=0.3, limit=1)
  if tags:
    tag = tags[0]
    tag.value = tag.value.strip().lstrip('№').lstrip().lstrip(':').lstrip('N ').lstrip().rstrip('.')
    nn_fix_span(tag)
    return tag


def nn_get_contract_date(textmap: TextMap, semantic_map: DataFrame) -> SemanticTag:
  tags = nn_get_tag_values('date', textmap, semantic_map, max_tokens=6, threshold=0.3, limit=1)
  if tags:
    tag = tags[0]
    _, dt = find_date(tag.value)
    tag.value = dt
    if dt is not None:
      return tag


def nn_get_tag_values(tagname: str, textmap: TextMap, semantic_map: pd.DataFrame, max_tokens, threshold=0.3, limit=1,
                      return_single=False) -> (SemanticTag or None) or [SemanticTag]:
  starts = semantic_map[tagname + "-begin"].values.copy()
  ends = semantic_map[tagname + "-end"].values.copy()

  starts[starts < threshold] = 0
  ends[ends < threshold] = 0

  #   print('ends argmax', ends.argmax())

  def top_inices(arr, limit):
    _tops = np.argsort(arr)[-limit:]
    return sorted(_tops)

  def next_index_from(b, ends):
    for e in sorted(ends):
      if e > b:
        if e - b > max_tokens:
          return b + max_tokens
        else:
          return e
    return -1

  def find_slices(begins, ends):

    for b in begins:
      e = next_index_from(b, ends)
      if e > 0:
        yield (b, e)

  def slice_confidence(sl, att):
    for s in sl:
      yield s, float(min(att[s[0]], att[s[1]]))

  top_starts = top_inices(starts, limit)
  top_ends = top_inices(ends, len(top_starts))

  slices = list(find_slices(top_starts, top_ends))
  slices = sorted(slice_confidence(slices, starts + ends), key=lambda x: x[1])[::-1]

  #   print('top_starts',top_starts)
  #   print('top_ends',top_ends)
  #   print('slices', slices)
  #   print()

  tags = []
  for s in slices:
    span = s[0]
    conf = s[1]
    if conf > 0:
      value = textmap.text_range(span)
      tag = SemanticTag(tagname, value, span)
      tag.confidence = conf  # float(att[slices[0]].mean())
      tags.append(tag)

  if return_single:
    if len(tags) > 0:
      return tags[0]
    else:
      return None

  return tags


def nn_fix_span(tag: SemanticTag):
  return tag
