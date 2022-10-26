import re
import warnings
from enum import Enum
from typing import Iterator

from pyjarowinkler import distance

from analyser.attributes import to_json
from analyser.case_numbers import find_case_number
from analyser.contract_agents import complete_re as agents_re, find_org_names, ORG_LEVELS_re, find_org_names_raw, \
  ContractAgent, _rename_org_tags, protocol_caption_complete_re, protocol_caption_complete_re_ignore_case
from analyser.doc_dates import find_document_date
from analyser.doc_numbers import document_number_c, find_document_number_in_subdoc
from analyser.documents import sentences_attention_to_words
from analyser.embedding_tools import AbstractEmbedder
from analyser.hyperparams import HyperParameters
from analyser.legal_docs import LegalDocument, ParserWarnings, \
  LegalDocumentExt
from analyser.ml_tools import SemanticTag, calc_distances_per_pattern_dict, max_exclusive_pattern_by_prefix, relu, \
  sum_probabilities, best_above, smooth_safe, spans_between_non_zero_attention, \
  FixedVector, estimate_confidence_by_mean_top_non_zeros, SemanticTagBase, max_confident_tag
from analyser.parsing import ParsingContext, AuditContext, find_value_sign_currency_attention
from analyser.patterns import AbstractPatternFactory, FuzzyPattern, create_value_negation_patterns, \
  create_value_patterns
from analyser.schemas import ProtocolSchema, AgendaItem, ContractPrice
from analyser.structures import ORG_LEVELS_names, OrgStructuralLevel
from analyser.text_normalize import r_group, r_quoted
from analyser.text_tools import is_long_enough, span_len
from analyser.transaction_values import complete_re as values_re

something = r'(\s*.{1,100}\s*)'
itog1 = r_group(r'\n' + r_group(r'итоги\s*голосования' + '|' + r'результаты\s*голосования') + r"[:\n]?")

_number_of_votes = r'(\s*[-: ]\s*)([0-9]|(нет)|[_]{1,10})[.;]*\s*'
r_votes_za = r_group(r_quoted('за') + _number_of_votes)
r_votes_pr = r_group(r_quoted('против') + _number_of_votes)
r_votes_vo = r_group(r_quoted('воздержался') + _number_of_votes)

protocol_votes_ = r_group(itog1 + something) + r_group(r_votes_za + r_votes_pr + r_votes_vo)
protocol_votes_re = re.compile(protocol_votes_, re.IGNORECASE | re.UNICODE)


class ProtocolAV(Enum):
  '''AV for Attention Vecotrs'''
  bin_votes_attention = 1
  relu_deal_approval = 2
  digits_attention = 3
  relu_value_attention_vector = 4


class ProtocolDocument(LegalDocumentExt):

  def __init__(self, doc: LegalDocument = None):
    super().__init__(doc)

    if doc is not None:
      # self.__dict__ = {**super().__dict__, **doc.__dict__}
      self.__dict__.update(doc.__dict__)

    self.attributes_tree = ProtocolSchema()

  def get_org_tags(self) -> [SemanticTag]:
    warnings.warn("please switch to attributes_tree struktur", DeprecationWarning)
    org = self.attributes_tree.org
    if org is not None:
      return _rename_org_tags([org], prefix="", start_from=1)

    return []

  org_tags = property(get_org_tags)

  def get_org_level(self) -> SemanticTagBase:
    return self.attributes_tree.structural_level

  def set_org_level(self, structural_level):
    self.attributes_tree.structural_level = structural_level

  org_level = property(get_org_level, set_org_level)

  def get_date(self) -> SemanticTagBase:
    return self.attributes_tree.date

  def set_date(self, date):
    self.attributes_tree.date = date

  date = property(get_date, set_date)

  def to_json_obj(self) -> dict:
    j: dict = super().to_json_obj()
    _attributes_tree_dict, _ = to_json(self.attributes_tree)
    j['attributes_tree'] = {"protocol": _attributes_tree_dict}
    return j


ProtocolDocument4 = ProtocolDocument  # aliasing #todo: remove it #still needed for old unit tests


class ProtocolParser(ParsingContext):
  patterns_dict = [
    ['sum_max1', 'стоимость не более 0 млн. тыс. миллионов тысяч рублей долларов копеек евро'],

    # ['solution_1','решение, принятое по вопросу повестки дня:'],
    # ['solution_2','по вопросам повестки дня приняты следующие решения:'],

    ['not_value_1', 'размер уставного капитала 0 рублей'],
    ['not_value_2', 'принятие решения о назначении секретаря'],

    ['agenda_end_1', 'кворум для проведения заседания и принятия решений имеется'],
    ['agenda_end_2', 'Вопрос повестки дня заседания'],
    ['agenda_end_3', 'Формулировка решения по вопросу повестки дня заседания:'],

    ['agenda_start_1', 'повестка дня заседания'],
    ['agenda_start_2', 'Повестка дня'],

    ['deal_approval_1', 'одобрить совершение сделки'],
    ['deal_approval_2', 'одобрить сделку'],
    ['deal_approval_3', 'дать согласие на заключение договора'],
    ['deal_approval_4', 'принять решение о совершении сделки'],
    ['deal_approval_5', 'принять решение о совершении крупной сделки'],
    ['deal_approval_6', 'заключить договор аренды'],
    ['deal_approval_7', 'Одобрить сделку, связанную с заключением Дополнительного соглашения'],

    ['question_1', 'По вопросу № 0'],
    ['question_2', 'Первый вопрос повестки дня заседания'],
    ['question_3', 'Решение, принятое по вопросу повестки дня:'],
    ['question_4', 'Решение, принятое по 1 вопросу повестки дня:'],

    ['footers_1', 'Время подведения итогов голосования'],
    ['footers_2', 'Список приложений:'],
    ['footers_3', 'Подсчет голосов производил Секретарь Совета директоров'],
    ['footers_4', 'Протокол составлен в 2-х экземплярах']

  ]

  def __init__(self, embedder: AbstractEmbedder = None, sentence_embedder: AbstractEmbedder = None):
    ParsingContext.__init__(self, embedder, sentence_embedder)

    self._protocols_factory: ProtocolPatternFactory or None = None
    self._patterns_embeddings = None

  def get_patterns_embeddings(self):
    if self._patterns_embeddings is None:
      patterns_te = [p[1] for p in ProtocolParser.patterns_dict]
      self._patterns_embeddings = self.get_sentence_embedder().embedd_strings(patterns_te)

    return self._patterns_embeddings

  def get_protocols_factory(self):
    if self._protocols_factory is None:
      self._protocols_factory = ProtocolPatternFactory(self.get_embedder())

    return self._protocols_factory

  def embedd(self, doc: ProtocolDocument):

    ### ⚙️🔮 SENTENCES embedding
    # if doc.sentence_map is None:
    #   doc.sentence_map = tokenize_doc_into_sentences_map(doc, HyperParameters.charter_sentence_max_len)
    doc.sentences_embeddings = self.get_sentence_embedder().embedd_strings(doc.sentence_map.tokens)

    ### ⚙️🔮 WORDS Embedding
    doc.embedd_tokens(self.get_embedder())

    doc.calculate_distances_per_pattern(self.get_protocols_factory())
    # TODO: should be made in analysys phase, not on embedding
    doc.distances_per_sentence_pattern_dict = calc_distances_per_pattern_dict(doc.sentences_embeddings,
                                                                              self.patterns_dict,
                                                                              self.get_patterns_embeddings())

  def find_org_date_number(self, doc: ProtocolDocument, ctx: AuditContext) -> ProtocolDocument:
    """
    phase 1, before embedding TF, GPU, and things
    searching for attributes required for filtering
    :param charter:
    :return:
    """
    # doc.sentence_map = tokenize_doc_into_sentences_map(doc, 250)

    doc.org_level = max_confident_tag(list(find_org_structural_level(doc)))
    doc.attributes_tree.org = find_protocol_org_obj(doc)
    doc.date = find_document_date(doc)
    doc.attributes_tree.case_number=find_case_number(doc)

    if doc.attributes_tree.org is not None:
      if doc.attributes_tree.org.name is None:
        doc.warn(ParserWarnings.org_name_not_found)

    if not doc.date:
      doc.warn(ParserWarnings.date_not_found)

    if not doc.org_level:
      doc.warn(ParserWarnings.org_struct_level_not_found)

    return doc

  def find_attributes(self, doc: ProtocolDocument, ctx: AuditContext = None) -> ProtocolDocument:

    self.find_org_date_number(doc, ctx)

    if doc.sentences_embeddings is None or doc.embeddings is None:
      self.embedd(doc)  # lazy embedding

    self.find_question_decision_sections(doc)
    self.find_margin_values(doc)
    self.find_contract_numbers(doc)
    self.find_contract_agents(doc, ctx.audit_subsidiary_name)

    self.validate(doc, ctx)
    return doc

  def validate(self, document: ProtocolDocument, ctx: AuditContext):

    if not document.attributes_tree.agenda_items:
      document.warn(ParserWarnings.protocol_agenda_not_found)

  def find_contract_agents(self,
                           doc: ProtocolDocument,
                           audit_subsidiary_name: str):

    for agenda_item in doc.attributes_tree.agenda_items:
      _subdoc = doc[agenda_item.slice]

      _all: [ContractAgent] = find_org_names_raw(_subdoc, max_names=10, parent=None, decay_confidence=False)
      all: [ContractAgent] = sorted(_all, key=lambda a: a.name.value != audit_subsidiary_name)

      # ai.contract.orgs = all
      for k, v in enumerate(all):
        agenda_item.get_contract_at(k).orgs.append(v)
        # TODO: one contract have several orgs!!

  def find_margin_values(self, doc):
    '''
    TODO: make doc.agenda_questions as the only argument, so it is independent of state
    :param doc:
    :return:
    '''
    if ProtocolAV.relu_value_attention_vector.name not in doc.distances_per_pattern_dict:
      raise KeyError('call find_question_decision_sections first')

    for agenda_item in doc.attributes_tree.agenda_items:
      agenda_subdoc = doc[agenda_item.as_slice()]
      relu_value_attention_vector = agenda_subdoc.distances_per_pattern_dict[
        ProtocolAV.relu_value_attention_vector.name]

      subdoc_values: [ContractPrice] = find_value_sign_currency_attention(agenda_subdoc,
                                                                          relu_value_attention_vector,
                                                                          parent_tag=agenda_item,
                                                                          absolute_spans=True)

      if len(subdoc_values) > 1:
        # decrease confidence
        confidence = 1.0 / len(subdoc_values)

        for k, v in enumerate(subdoc_values):
          v *= confidence
          # v.parent.kind = SemanticTag.number_key(v.parent.kind, k)

      for k, v in enumerate(subdoc_values):
        agenda_item.get_contract_at(k).price = v

  def find_contract_numbers(self, doc):

    for agenda_item in doc.attributes_tree.agenda_items:
      subdoc = doc[agenda_item.slice]

      numbers: [SemanticTag] = find_document_number_in_subdoc(subdoc, tagname='number',
                                                              parent=None)

      for k, v in enumerate(numbers):
        agenda_item.get_contract_at(k).number = v

  def collect_spans_having_votes(self, segments, textmap):
    warnings.warn("use TextMap.regex_attention", DeprecationWarning)
    """
    search for votes in each document segment
    collect only
    :param segments:
    :param textmap:
    :return:  segments with votes
    """
    for span in segments:
      subdoc = textmap.slice(span)
      protocol_votes = list(subdoc.finditer(protocol_votes_re))
      if protocol_votes:
        yield span

  def find_protocol_sections_edges(self, distances_per_pattern_dict):

    patterns = ['footers_', 'deal_approval_', 'question_']
    vv_ = []
    for p in patterns:
      v_ = max_exclusive_pattern_by_prefix(distances_per_pattern_dict, p)
      v_ = relu(v_, 0.55)
      vv_.append(v_)

    v_sections_attention = sum_probabilities(vv_)
    v_sections_attention = best_above(v_sections_attention, 0.55)
    return v_sections_attention

  def _get_value_attention_vector(self, doc: LegalDocument):
    s_value_attention_vector = max_exclusive_pattern_by_prefix(doc.distances_per_pattern_dict, 'sum_max_p_')

    doc.distances_per_pattern_dict['__max_value_av'] = s_value_attention_vector  # just for debugging
    not_value_av = max_exclusive_pattern_by_prefix(doc.distances_per_pattern_dict, 'not_sum_')

    not_value_av = smooth_safe(not_value_av, window_len=5)

    not_value_av = relu(not_value_av, 0.5)
    doc.distances_per_pattern_dict['__not_value_av'] = not_value_av  # just for debugging

    s_value_attention_vector -= not_value_av * 0.8
    s_value_attention_vector = relu(s_value_attention_vector, 0.3)
    return s_value_attention_vector

  def find_question_decision_sections(self, doc: ProtocolDocument) -> [AgendaItem]:

    # DEAL APPROVAL SENTENCES
    _v_deal_approval: FixedVector = max_exclusive_pattern_by_prefix(doc.distances_per_sentence_pattern_dict,
                                                                    'deal_approval_')
    _spans, deal_approval_av = sentences_attention_to_words(_v_deal_approval, doc.sentence_map, doc.tokens_map)
    deal_approval_relu_av: FixedVector = best_above(deal_approval_av, 0.5)

    # VOTES
    votes_av: FixedVector = doc.tokens_map.regex_attention(protocol_votes_re)

    # DOC NUMBERS
    numbers_av: FixedVector = doc.tokens_map.regex_attention(document_number_c)

    # DOC AGENTS orgs
    agents_av: FixedVector = doc.tokens_map.regex_attention(agents_re)

    # DOC MARGIN VALUES
    margin_values_av: FixedVector = self._get_value_attention_vector(doc)
    margin_values_v: FixedVector = doc.tokens_map.regex_attention(values_re)
    margin_values_v *= margin_values_av
    doc.distances_per_pattern_dict[ProtocolAV.relu_value_attention_vector.name] = margin_values_av

    # -----
    combined_av: FixedVector = sum_probabilities([
      deal_approval_relu_av,
      margin_values_v,
      agents_av / 2,
      votes_av / 2,
      numbers_av / 2])

    # TODO: this is exactly what we will get from NN out layer
    combined_av_norm: FixedVector = best_above(combined_av, 0.2)
    # --------------

    _protocol_sections_edges = self.find_protocol_sections_edges(doc.distances_per_sentence_pattern_dict)
    _question_spans_sent = spans_between_non_zero_attention(_protocol_sections_edges)
    question_spans_words = doc.sentence_map.remap_slices(_question_spans_sent, doc.tokens_map)

    _agenda_questions_iter = find_confident_spans(question_spans_words, combined_av_norm, 'agenda_item', 0.5)

    for aq in _agenda_questions_iter:
      agenda_item = AgendaItem(tag=aq)
      doc.attributes_tree.agenda_items.append(agenda_item)

    return doc.attributes_tree.agenda_items


class ProtocolPatternFactory(AbstractPatternFactory):

  def create_pattern(self, pattern_name, ppp):
    _ppp = (ppp[0].lower(), ppp[1].lower(), ppp[2].lower())
    fp = FuzzyPattern(_ppp, pattern_name)
    self.patterns.append(fp)
    return fp

  def __init__(self, embedder):
    AbstractPatternFactory.__init__(self)

    create_value_negation_patterns(self)
    create_value_patterns(self)

    self.embedd(embedder)


def find_protocol_org_obj(protocol: ProtocolDocument) -> ContractAgent or None:
  _subdoc = protocol[0:HyperParameters.protocol_caption_max_size_words]
  orgs: [ContractAgent] = find_org_names_raw(_subdoc, max_names=1, regex=protocol_caption_complete_re,
                                             re_ignore_case=protocol_caption_complete_re_ignore_case)
  if len(orgs) == 0:
    return None

  return orgs[0]


def find_protocol_org(protocol: ProtocolDocument) -> [SemanticTag]:
  warnings.warn("please switch to find_protocol_org_obj", DeprecationWarning)

  ret = []
  _subdoc = protocol[0:HyperParameters.protocol_caption_max_size_words]
  _flat_list: [SemanticTag] = find_org_names(_subdoc,
                                             max_names=1,
                                             regex=protocol_caption_complete_re,
                                             re_ignore_case=protocol_caption_complete_re_ignore_case)

  nm = SemanticTag.find_by_kind(_flat_list, 'org-1-name')
  if nm is not None:
    ret.append(nm)
  else:
    protocol.warn(ParserWarnings.org_name_not_found)

  tp = SemanticTag.find_by_kind(_flat_list, 'org-1-type')
  if tp is not None:
    ret.append(tp)
  else:
    protocol.warn(ParserWarnings.org_type_not_found)
  return ret


def closest_name(pattern: str, knowns: [str]) -> (str, int):
  #
  min_distance = 0
  found = None
  for b in knowns:
    d = distance.get_jaro_distance(pattern, b, winkler=True, scaling=0.1)
    if d > min_distance:
      found = b
      min_distance = d

  return found, min_distance


def find_org_structural_level(doc: LegalDocument) -> Iterator[SemanticTag]:
  compiled_re = re.compile(ORG_LEVELS_re, re.MULTILINE | re.IGNORECASE | re.UNICODE)

  entity_type = 'org_structural_level'
  for m in re.finditer(compiled_re, doc.text):

    char_span = m.span(entity_type)
    span = doc.tokens_map.token_indices_by_char_range(char_span)
    val = doc.tokens_map.text_range(span)

    val, conf = closest_name(val, ORG_LEVELS_names)

    confidence = conf * (1.0 - (span[0] / len(doc)))  # relative distance from the beginning of the document
    if span_len(char_span) > 1 and is_long_enough(val):

      if confidence > HyperParameters.org_level_min_confidence:
        tag = SemanticTag(entity_type, OrgStructuralLevel.find_by_display_string(val), span)
        tag.confidence = confidence

        yield tag


def find_confident_spans(slices: [int], block_confidence: FixedVector, tag_name: str, threshold: float) -> Iterator[
  SemanticTag]:
  count = 0
  for _slice in slices:
    pv = block_confidence[_slice[0]:_slice[1]]
    confidence = estimate_confidence_by_mean_top_non_zeros(pv, 5)

    if confidence > threshold:
      count += 1
      st = SemanticTag(SemanticTag.number_key(tag_name, count), None, _slice)
      st.confidence = confidence
      yield (st)
