from enum import Enum

from overrides import overrides
from pandas import DataFrame

from analyser.attributes import to_json
from analyser.contract_agents import ContractAgent, normalize_contract_agent
from analyser.doc_dates import find_date
from analyser.documents import TextMap
from analyser.hyperparams import HyperParameters
from analyser.insides_finder import InsidesFinder
from analyser.legal_docs import LegalDocument, ContractValue, ParserWarnings
from analyser.log import logger
from analyser.ml_tools import SemanticTag, SemanticTagBase, is_span_intersect
from analyser.parsing import ParsingContext, AuditContext, find_value_sign_currency_attention
from analyser.patterns import AV_SOFT, AV_PREFIX
from analyser.schemas import ContractSchema, OrgItem, ContractPrice
from analyser.text_normalize import r_human_name_compilled
from analyser.text_tools import find_top_spans
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

    # if not contract.attributes_tree.number:
    #   contract.attributes_tree.number = nn_get_contract_number(_contract_cut.tokens_map, semantic_map)
    #
    # if not contract.date:
    #   contract.date = nn_get_contract_date(_contract_cut.tokens_map, semantic_map)
    #
    # # -------------------------------
    # # -------------------------------orgs, agents
    # if (contract.attributes_tree.orgs is None) or len(contract.attributes_tree.orgs) < 2:
    #   contract.attributes_tree.orgs = nn_find_org_names(_contract_cut.tokens_map, semantic_map,
    #                                                     audit_ctx=ctx)
    #
    # check_orgs_natural_person(contract.attributes_tree.orgs, contract.get_headline())  # mutator

    # -------------------------------subject
    contract.subject = nn_get_subject(_contract_cut.tokens_map, semantic_map, subj_1hot)

    # -------------------------------values, prices, amounts
    self._logstep("finding contract values")
    contract.contract_values = nn_find_contract_value(_contract_cut, semantic_map)

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
  for o in [1, 2]:
    ca: ContractAgent = ContractAgent()
    for n in ['name', 'alias', 'type']:
      tagname = f'org-{o}-{n}'
      tag = nn_get_tag_value(tagname, textmap, semantic_map)
      setattr(ca, n, tag)
    normalize_contract_agent(ca)
    contract_agents.append(ca)

  # filtrationd
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


def nn_find_contract_value(contract: ContractDocument, semantic_map: DataFrame) -> [ContractPrice]:
  _keys = ['sign_value_currency/value', 'sign_value_currency/currency', 'sign_value_currency/sign']
  attention_vector = semantic_map[_keys].values.sum(axis=-1)

  values_list: [ContractPrice] = find_value_sign_currency_attention(contract, attention_vector)
  if len(values_list) == 0:
    return []
  # ------
  # reduce number of found values
  # take only max value and most confident ones (we hope, it is the same finding)

  max_confident_cv: ContractPrice = max_confident(values_list)
  if max_confident_cv.amount.confidence < 0.1:
    return []

  return [max_confident_cv]

  # max_valued_cv: ContractValue = max_value(values_list)
  # if max_confident_cv == max_valued_cv:
  #   return [max_confident_cv]
  # else:
  #   # TODO: Insurance docs have big value, its not what we're looking for. Biggest is not the best see https://github.com/nemoware/analyser/issues/55
  #   # TODO: cannot compare diff. currencies
  #   max_valued_cv *= 0.5
  #   return [max_valued_cv]


def nn_get_subject(textmap: TextMap, semantic_map: DataFrame, subj_1hot) -> SemanticTag:
  predicted_subj_name, confidence, _ = decode_subj_prediction(subj_1hot)

  # tag = SemanticTag(None, predicted_subj_name.name, span=None)

  tag_ = nn_get_tag_value('subject', textmap, semantic_map)
  span = None
  if tag_ is not None:
    span = tag_.span
  tag = SemanticTag(None, predicted_subj_name.name, span=span, confidence=confidence)

  return tag


def nn_get_contract_number(textmap: TextMap, semantic_map: DataFrame) -> SemanticTag:
  tag = nn_get_tag_value('number', textmap, semantic_map)
  if tag is not None:
    tag.value = tag.value.strip().lstrip('№').lstrip().lstrip(':').lstrip('N ').lstrip().rstrip('.')
    nn_fix_span(tag)
  return tag


def nn_get_contract_date(textmap: TextMap, semantic_map: DataFrame) -> SemanticTag:
  tag = nn_get_tag_value('date', textmap, semantic_map)
  if tag is not None:
    _, dt = find_date(tag.value)
    tag.value = dt
    if dt is not None:
      return tag


def nn_get_tag_value(tagname: str, textmap: TextMap, semantic_map: DataFrame, threshold=0.3) -> SemanticTag or None:
  att = semantic_map[tagname].values
  slices = find_top_spans(att, threshold=threshold, limit=1)  # TODO: estimate per-tag thresholds

  if len(slices) > 0:
    span = slices[0].start, slices[0].stop
    value = textmap.text_range(span)
    tag = SemanticTag(tagname, value, span)
    tag.confidence = float(att[slices[0]].mean())
    return tag
  return None


def nn_fix_span(tag: SemanticTag):
  return tag
