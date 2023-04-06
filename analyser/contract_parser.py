from enum import Enum

import pandas as pd
from overrides import overrides
from pandas import DataFrame

from analyser.attributes import to_json
from analyser.case_numbers import find_case_number
from analyser.contract_agents import ContractAgent, normalize_contract_agent
from analyser.doc_dates import find_date
from analyser.documents import TextMap
from analyser.hyperparams import HyperParameters
from analyser.insides_finder import InsidesFinder
from analyser.legal_docs import LegalDocument, ContractValue, ParserWarnings, find_value_sign
from analyser.log import logger
from analyser.ml_tools import SemanticTag, SemanticTagBase, is_span_intersect
from analyser.parsing import ParsingContext, AuditContext
from analyser.patterns import AV_SOFT, AV_PREFIX
from analyser.schemas import ContractSchema, OrgItem, ContractPrice, merge_spans
from analyser.text_normalize import r_human_name_compilled
from analyser.text_tools import to_float, span_len
from analyser.transaction_values import ValueSpansFinder
from gpn.gpn import is_gpn_name
from tf_support.tf_subject_model import load_subject_detection_trained_model, decode_subj_prediction, nn_predict


class ContractDocument(LegalDocument):

  def __init__(self, original_text):
    LegalDocument.__init__(self, original_text)

    self.attributes_tree = ContractSchema()

  def to_json_obj(self) -> dict:
    j: dict = super().to_json_obj()
    _attributes_tree_dict, _ = to_json(self.attributes_tree)
    j['attributes_tree']['contract'] = _attributes_tree_dict
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


class GenericParser(ParsingContext):
  def __init__(self, embedder=None, sentence_embedder=None):
    ParsingContext.__init__(self, embedder, sentence_embedder)

  def find_org_date_number(self, doc: LegalDocument, ctx: AuditContext) -> LegalDocument:
    doc.attributes_tree.case_number = find_case_number(doc)
    return doc

  def find_attributes(self, d: LegalDocument, ctx: AuditContext) -> LegalDocument:
    self._reset_context()
    d = self.find_org_date_number(d, ctx)
    return d


class ContractParser(GenericParser):
  def __init__(self, embedder=None, sentence_embedder=None):
    ParsingContext.__init__(self, embedder, sentence_embedder)
    self.subject_prediction_model = load_subject_detection_trained_model()
    self.insides_finder = InsidesFinder()

  def find_org_date_number(self, contract: ContractDocument, ctx: AuditContext) -> ContractDocument:

    # GenericParser is called an all documents before this
    # super().find_org_date_number(contract, ctx)

    _head = contract[0:300]  # warning, trimming doc for analysis phase 1
    if _head.embeddings is None:
      logger.debug('embedding 300-trimmed contract')
      _head.embedd_tokens(self.get_embedder())

    # predicting with NN
    logger.debug('predicting semantic_map in 300-trimmed contract with NN')
    semantic_map, _ = nn_predict(self.subject_prediction_model, _head)

    contract.attributes_tree.orgs = nn_find_org_names(_head.tokens_map, semantic_map,
                                                      audit_ctx=ctx)
    check_orgs_natural_person(contract.attributes_tree.orgs, contract.get_headline(), ctx)  # mutator

    # TODO: maybe move contract.tokens_map into text map
    # contract.attributes_tree.case_number = find_case_number(contract)
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
      _contract_cut = contract[0:HyperParameters.max_doc_size_tokens]  # warning, trimming doc for analysis phase 1

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

    check_orgs_natural_person(contract.attributes_tree.orgs, contract.get_headline(), ctx)  # mutator

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
    #     self.insides_finder.find_insides(contract)

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
  # TODO:SORT ORDER, see 63c506cbe2456d59975e12a6
  contract_agents: [ContractAgent] = []

  types = nn_get_tag_values('org-type', textmap, semantic_map, max_tokens=12, threshold=0.5,
                            limit=2)
  names = nn_get_tag_values('org-name', textmap, semantic_map, max_tokens=12, threshold=0.5,
                            limit=2)
  aliases = nn_get_tag_values('org-alias', textmap, semantic_map, max_tokens=4, threshold=0.5,
                              limit=2)

  _list = types + names + aliases
  _list = sorted(_list, key=lambda x: x.span[0])

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

  check_org_intersections(contract_agents)  # mutator

  return contract_agents  # _swap_org_tags(cas)


def check_orgs_natural_person(contract_agents: [OrgItem], header0: str, ctx: AuditContext):
  if not contract_agents:
    return

  for contract_agent in contract_agents:
    check_org_is_natural_person(contract_agent, ctx)

  if header0:
    if header0.lower().find('с физическим лицом') >= 0:
      _set_natural_person(contract_agents[-1])
      # TODO: why setting it to the last array element??

  return contract_agents


def check_org_is_natural_person(contract_agent: OrgItem, audit_ctx: AuditContext):
  human_name = False

  if contract_agent.type is not None:
    if len(contract_agent.type) >= 2:
      return

  if contract_agent.name is not None:
    name: str = contract_agent.name.value

    if contract_agent.is_known_subsidiary:
      # known subsidiary may not be natural person
      return

    if is_gpn_name(name):  # TODO: hack
      return

    if audit_ctx.is_same_org(name):
      # known subsidiary may not be natural person
      return

    x = r_human_name_compilled.search(name)

    if x is not None:
      human_name = True

  if human_name:
    _set_natural_person(contract_agent)


def _set_natural_person(contract_agent: OrgItem):
  if contract_agent.type is None:
    contract_agent.type = SemanticTag(None, None)

  contract_agent.type.value = 'Физическое лицо'


def check_org_intersections(contract_agents: [OrgItem]) -> None:
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

  # return contract_agents


def nn_find_contract_value(textmap: TextMap, tagsmap: DataFrame) -> [ContractPrice]:
  # TODO: FIX SENTENCE!

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
  cp.vat_unit = nn_get_tag_values('vat_unit', textmap, tagsmap, max_tokens=3, threshold=0.02, limit=1,
                                  return_single=True)

  sentence_seed_tag = None
  _order = [cp.amount_netto, cp.amount_brutto, cp.amount]
  for t in _order:
    if t is not None:
      sentence_seed_tag = t
      break

  if not sentence_seed_tag:
    return []

  def fix_parent_span():
    cp.span = merge_spans(cp.list_children())
    if parent_tag:
      cp.span = merge_spans([cp, parent_tag])

    if span_len(cp.span) > 200 or span_len(cp.span) < 20:
      sentence_span = textmap.sentence_at_index(sentence_seed_tag.span[0])
      cp.span = sentence_span

  #   ///SIGN

  if cp.sign:
    _start = cp.sign.span[0] - 5
    region = textmap.slice(slice(_start, cp.sign.span[0] + 5))
    _sign, _sign_span = find_value_sign(region)
    cp.sign.value = _sign
    if _sign_span is not None:
      cp.sign.span = _sign_span
      cp.sign.offset(_start)

  fix_parent_span()

  try:
    region = textmap.text_range(cp.span)
    region_map = TextMap(region)
    results = ValueSpansFinder(region)

    # if results.including_vat:
    cp.amount = SemanticTag('amount')
    cp.amount.span = region_map.token_indices_by_char_range(results.number_span)
    cp.amount.value = results.original_sum
    cp.amount.offset(cp.span[0])
    if results.including_vat == False:
      cp.amount.value = results.value

    if cp.currency is None:
      cp.currency = SemanticTag('currency')
      cp.currency.span = region_map.token_indices_by_char_range(results.currency_span)
      cp.currency.offset(cp.span[0])
    cp.currency.value = results.currencly_name


  except TypeError as e:
    logger.exception(f'smthinf wrong {str(cp)=}')
    logger.error(e)
    results = None

  if cp.amount:
    try:
      cp.amount.value = to_float(cp.amount.value)
    except Exception as e:
      logger.error(f'amount is {cp.amount}')
    # logger.error(e)

  if cp.vat:
    try:
      cp.vat.value = to_float(cp.vat.value)
    except Exception as e:
      # logger.error(e)
      logger.error(f'vat is {cp.vat.value}, cannot cast to float')

  if cp.amount_netto:
    try:
      cp.amount_netto.value = to_float(cp.amount_netto.value)
    except Exception as e:
      # logger.error(e)
      logger.error(f'amount_netto is {cp.amount_netto}')

  if cp.amount_brutto:
    try:
      cp.amount_brutto.value = to_float(cp.amount_brutto.value)
    except Exception as e:
      logger.error(f'amount_brutto is {cp.amount_brutto}')

  if (results is not None):
    if results.including_vat:
      cp.amount_brutto = cp.amount
      cp.amount_netto = None
    else:
      cp.amount_netto = cp.amount
      cp.amount_brutto = None
    cp.amount = None

  fix_parent_span()
  return [cp]


def nn_get_subject(textmap: TextMap, semantic_map: DataFrame, subj_1hot) -> SemanticTag:
  # TODO: FIX SENTENCE!
  predicted_subj_name, confidence, _ = decode_subj_prediction(subj_1hot)

  tags = nn_get_tag_values('subject', textmap, semantic_map, max_tokens=200, threshold=0.02,
                           limit=1)

  span = None
  if tags:
    tag = tags[0]
    span = tag.span
    if span_len(span) < 30 or span_len(span) > 150:
      # TODO: FIX SENTENCE!
      sentence_span = textmap.sentence_at_index(tag.span[0])
      span = sentence_span

  tag = SemanticTag(None, predicted_subj_name.name, span=span, confidence=confidence)

  return tag


def fix_contract_number(tag: SemanticTag, textmap: TextMap) -> SemanticTag or None:
  if tag:
    span = [tag.span[0], tag.span[1]]
    for i in range(tag.span[0], tag.span[1]):
      if i < 0 or i >= len(textmap):
        msg = f'{i=} {len(textmap)=} {str(tag)=} {tag.span=}'
        logger.error(msg)
        raise ValueError(msg)

      t = textmap[i]
      t = t.strip().lstrip('№').lstrip().lstrip(':').lstrip('N ').lstrip().rstrip('.').rstrip('_').lstrip('_')
      if t == '':
        span[0] = i + 1
    tag.span = span
  if span_len(tag.span) == 0:
    return None

  return tag


def nn_get_contract_number(textmap: TextMap, semantic_map: DataFrame) -> SemanticTag:
  tags = nn_get_tag_values('number', textmap, semantic_map, max_tokens=5, threshold=0.3, limit=1)
  if tags:
    tag = tags[0]
    tag.value = tag.value.strip().lstrip('№').lstrip().lstrip(':').lstrip('N ').lstrip().rstrip('.').rstrip('_').lstrip(
      '_')
    if tag.value == '':
      return None
    tag = fix_contract_number(tag, textmap)
    return tag


def nn_get_contract_date(textmap: TextMap, semantic_map: DataFrame) -> SemanticTag:
  tag_name = 'date'
  date_index = semantic_map[f'{tag_name}-begin'].argmax()
  confidence = float(semantic_map[f'{tag_name}-begin'][date_index])
  sentense_span = textmap.sentence_at_index(date_index)
  date_sentence = textmap.text_range(sentense_span)

  #       print(f'{date_sentence=}')

  _charspan, dt = find_date(date_sentence)
  if dt is not None and confidence > 0.01:  # TODO:
    region_map = TextMap(date_sentence)
    span = region_map.token_indices_by_char_range(_charspan)
    span = span[0] + sentense_span[0], span[1] + sentense_span[0]
    tag = SemanticTag(tag_name, dt, span)
    tag.confidence = confidence
    return tag
  return None


def nn_get_contract_date_OLD(textmap: TextMap, semantic_map: DataFrame) -> SemanticTag:
  tags = nn_get_tag_values('date', textmap, semantic_map, max_tokens=6, threshold=0.3, limit=1)
  if tags:
    tag = tags[0]
    _, dt = find_date(tag.value)
    tag.value = dt
    if dt is not None:
      return tag


def nn_get_tag_values(tag_name: str,
                      textmap: TextMap,
                      tagsmap: pd.DataFrame,
                      max_tokens=200,
                      threshold=0.3,  # TODO: what's that
                      limit=1,
                      return_single=False) -> (SemanticTag or None) or [SemanticTag]:
  if len(textmap) < 1:
    return None

  attention = tagsmap[tag_name + '-begin'][:len(textmap)].values.copy()

  threshold = max(attention.max() * 0.8, 0.043)

  last_taken = False
  sequences = []
  seq = None

  # collecting hits--------
  for i, v in enumerate(attention):
    if v >= threshold:
      #             print ('---',i,f'{v:.2}', a_doc_from_json.get_tokens_map_unchaged()[i])
      if seq is None:
        seq = []
        sequences.append(seq)
      seq.append(i)
      last_taken = True
    else:
      if last_taken:
        seq = None
        last_taken = False

  # making spans  --------
  tags = []
  for s in sequences:
    span = [min(s), max(s) + 1]
    if span[1] - span[0] > max_tokens:
      span[1] = min(len(textmap) - 1, span[0] + max_tokens)

    if span[1] - span[0] > 0:
      quote = textmap.text_range(span)
      tag = SemanticTag(tag_name, quote, span)
      tag.confidence = float(attention[span[0]:span[1]].mean())

      #         print(span, quote, tag)
      tags.append(tag)

  # sorting spans--------
  tags = sorted(tags, key=lambda x: -x.confidence)
  tags = tags[0:limit]

  if return_single:
    if len(tags) > 0:
      return tags[0]
    else:
      return None

  tags = sorted(tags, key=lambda x: x.span[0])

  return tags


def nn_fix_span(tag: SemanticTag):
  # TODO:MAKE IT HAPPEN
  return tag
