import html as escaper
from typing import List

import numpy as np

from legal_docs import CharterDocument, ProtocolDocument
from ml_tools import ProbableValue
from parsing import known_subjects, head_types_dict
from patterns import AV_PREFIX, AV_SOFT, PatternSearchResult, ConstraintsSearchResult, PatternSearchResults
from structures import ContractSubject
from structures import OrgStructuralLevel
from transaction_values import ValueConstraint

head_types_colors = {'head.directors': 'crimson',
                     'head.all': 'orange',
                     'head.gen': 'blue',
                     'head.shareholders': '#666600',
                     'head.pravlenie': '#0099cc',
                     'head.unknown': '#999999'}

org_level_colors = {OrgStructuralLevel.BoardOfDirectors: 'crimson',
                    OrgStructuralLevel.ShareholdersGeneralMeeting: 'orange',
                    OrgStructuralLevel.CEO: 'blue',
                    OrgStructuralLevel.BoardOfCompany: '#0099cc',
                    None: '#999999'}

known_subjects_dict = {
  ContractSubject.Charity: 'Благотворительность',
  ContractSubject.RealEstate: "Сделки с имуществом",
  ContractSubject.Lawsuit: "Судебные споры",
  ContractSubject.Deal: "Совершение сделки",
  ContractSubject.Other: "Прочее"
}

org_level_dict = {OrgStructuralLevel.BoardOfDirectors: 'Совет директоров',
                  OrgStructuralLevel.ShareholdersGeneralMeeting: 'Общее собрание участников/акционеров',
                  OrgStructuralLevel.CEO: 'Генеральный директор',
                  OrgStructuralLevel.BoardOfCompany: 'Правление общества',
                  None: '*Неизвестный орган управления*'}

WARN = '\033[1;31m======== Dear Artem, ACHTUNG! 🔞 '


def as_smaller(x):
  return f'<span style="font-size:80%;">{x}</span>'


def as_error_html(txt):
  return f'<div style="color:red">⚠️ {txt}</div>'


def as_warning(txt):
  return f'<div style="color:orange">⚠️ {txt}</div>'


def as_msg(txt):
  return f'<div>{txt}</div>'


def as_quote(txt):
  return f'<i style="margin-top:0.2em; margin-left:2em; font-size:90%">"...{txt} ..."</i>'


def as_headline_2(txt):
  return f'<h2>{txt}</h2>'


def as_headline_3(txt):
  return f'<h3 style="margin:0">{txt}</h3>'


def as_headline_4(txt):
  return f'<h4 style="margin:0">{txt}</h4>'


def as_offset(txt):
  return f'<div style="margin-left:2em">{txt}</div>'


def as_currency(v):
  if v is None: return "any"
  return f'{v.value:20,.0f} {v.currency} '


class AbstractRenderer:

  def sign_to_text(self, sign: int):
    if sign < 0: return " < "
    if sign > 0: return " > "
    return ' = '

  def sign_to_html(self, sign: int):
    if sign < 0: return " &lt; "
    if sign > 0: return " &gt; "
    return ' = '

  def value_to_html(self, vc: ValueConstraint):
    color = '#333333'
    if vc.sign > 0:
      color = '#993300'
    elif vc.sign < 0:
      color = '#009933'

    return f'<b style="color:{color}">{self.sign_to_html(vc.sign)} {vc.currency} {vc.value:20,.2f}</b> '

  def render_value_section_details(self, value_section_info):
    pass

  def to_color_text(self, tokens, weights, colormap='coolwarm', print_debug=False, _range=None) -> str:
    pass

  def render_color_text(self, tokens, weights, colormap='coolwarm', print_debug=False, _range=None):
    pass

  def print_results(self, doc, results):
    raise NotImplementedError()

  def render_values(self, values: List[ProbableValue]):
    for pv in values:
      vc = pv.value
      s = f'{self.sign_to_text(vc.sign)} \t {vc.currency} \t {vc.value:20,.2f} \t {pv.confidence:20,.2f} '
      print(s)

  def render_contents(self, doc):
    pass


class SilentRenderer(AbstractRenderer):
  pass


v_color_map = {
  'deal_value_attention_vector': (1, 0.0, 0.5),
  'soft$.$at_sum__': (0.9, 0.5, 0.0),

  '$at_sum__': (0.9, 0, 0.1),
  'soft$.$at_d_order_': (0.0, 0.3, 0.9),

  f'{AV_PREFIX}margin_value': (1, 0.0, 0.5),
  f'{AV_SOFT}{AV_PREFIX}margin_value': (1, 0.0, 0.5),

  f'{AV_PREFIX}x_{ContractSubject.Charity}': (0.0, 0.9, 0.3),
  f'{AV_SOFT}{AV_PREFIX}x_{ContractSubject.Charity}': (0.0, 1.0, 0.0),

  f'{AV_PREFIX}x_{ContractSubject.Lawsuit}': (0.8, 0, 0.7),
  f'{AV_SOFT}{AV_PREFIX}x_{ContractSubject.Lawsuit}': (0.9, 0, 0.9),

  f'{AV_PREFIX}x_{ContractSubject.RealEstate}': (0.2, 0.2, 1),
  f'{AV_SOFT}{AV_PREFIX}x_{ContractSubject.RealEstate}': (0.2, 0.2, 1),
}

colors_by_contract_subject = {
  ContractSubject.RealEstate: (0.2, 0.2, 1),
  ContractSubject.Lawsuit: (0.9, 0, 0.9),
  ContractSubject.Charity: (0.0, 0.9, 0.3),
}

for k in colors_by_contract_subject:
  v_color_map[f'{AV_SOFT}{AV_PREFIX}x_{k}'] = colors_by_contract_subject[k]


class HtmlRenderer(AbstractRenderer):
  ''' AZ:-Rendering CHARITY🔥-----💸------💸-------💸------------------------------'''

  def html_charity_constraints_by_head(self, charity_constraints_by_head: PatternSearchResults) -> str:
    html = ''

    html += as_headline_3('одобрение сделок Благотворительности:')

    if len(charity_constraints_by_head) > 0:

      for r in charity_constraints_by_head:
        html += '<BR>'

        v = {
          'soft$.' + r.attention_vector_name: r.parent.distances_per_pattern_dict['soft$.' + r.attention_vector_name],
          r.attention_vector_name: r.parent.distances_per_pattern_dict[r.attention_vector_name],
        }

        min_color = (0.3, 0.3, 0.33)
        q_html = ''
        q_html += to_multicolor_text(r.parent.tokens_cc, v,
                                     v_color_map, min_color=min_color, _slice=r.region)
        html += as_c_quote(q_html)
    else:
      html += as_msg('не выявлено')

    return html

  def render_constraint_values(self, doc, rz, charity_constraints):

    print(WARN + f'{self.render_constraint_values} is deprecated, use {self.render_constraint_values_2}')
    from legal_docs import ConstraintsSearchResult

    html = ''
    for head_type in rz.keys():

      constraint_search_results: List[ConstraintsSearchResult] = rz[head_type]

      html += '<hr style="margin-top: 45px">'

      html += f'<h2 style="color:{head_types_colors[head_type]}; padding:0; margin:0">{head_types_dict[head_type]}</h2>'

      # html += as_quote(r_by_head_type.)

      charity_constraints_by_head = charity_constraints[head_type]

      html_i = ''
      html_i += self.html_charity_constraints_by_head(charity_constraints_by_head)

      if True:
        html_i += as_headline_3('Решения о пороговых суммах:')

        if len(constraint_search_results) > 0:
          for constraint_search_result in constraint_search_results:
            html_i += self._render_sentence(constraint_search_result)

        else:
          html_i += as_error_html('Пороговые суммы не найдены или не заданы')

      html += as_offset(html_i)

    return html

  def render_constraint_values_2(self, charter: CharterDocument) -> str:
    from structures import OrgStructuralLevel

    html = ''

    for level in OrgStructuralLevel:

      constraint_search_results: List[PatternSearchResult] = charter.constraints_by_org_level(level)

      html += '<hr style="margin-top: 45px">'

      html += f'<h2 style="color:{org_level_colors[level]}; ' \
              f'padding:0; margin:0">{org_level_dict[level]}</h2>'

      html_i = ''
      html_i += as_headline_3('Компетенции (и пороговые суммы):')

      rendered = 0
      for constraint_search_result in constraint_search_results:
        rendered += 1
        html_i += self.constraints_to_html(constraint_search_result)

      if rendered == 0:
        html_i += as_warning('Ограничения и пороговые суммы не обнаружены или не заданы')

      html += as_offset(html_i)

    return html

  def _to_color_text(self, _tokens, weights, mpl, colormap='coolwarm', _range=None, separator=' '):
    tokens = [escaper.escape(t) for t in _tokens]

    if len(tokens) == 0:
      return " - empty -"

    if len(weights) != len(tokens):
      raise ValueError("number of weights differs weights={} tokens={}".format(len(weights), len(tokens)))

    #   if()
    vmin = weights.min()
    vmax = weights.max()

    if _range is not None:
      vmin = _range[0]
      vmax = _range[1]

    norm = mpl.colors.Normalize(vmin=vmin - 0.5, vmax=vmax)
    cmap = mpl.cm.get_cmap(colormap)

    html = ""

    for d in range(0, len(weights)):
      word = tokens[d]
      if word == ' ':
        word = '&nbsp;_ '
      token_color = mpl.colors.to_hex(cmap(norm(weights[d])))
      html += f'<span title="{d} {weights[d]:.4f}" style="background-color:{token_color}">{word}{separator}</span>'

      if tokens[d] == '\n':
        html += "¶<br>"

    return html

  def constraints_to_html(self, search_result: PatternSearchResult):

    constraints: List[ValueConstraint] = search_result.constraints

    html = ""
    html += "<br>"
    html += '<div style="border-bottom:1px solid #ccc; margin-top:1em"></div>'

    subj_type = search_result.subject_mapping["subj"]
    if subj_type in known_subjects_dict:
      subj_type = known_subjects_dict[subj_type]
    html += as_headline_4(f'{subj_type} <sup>confidence={search_result.subject_mapping["confidence"]:.3f}')

    for probable_v in constraints:
      html += self.value_to_html(probable_v.value)

    attention_vectors = self.map_attention_vectors_to_colors(search_result)

    html += as_offset(as_c_quote(to_multicolor_text(search_result.tokens, attention_vectors,
                                                    v_color_map,
                                                    min_color=(0.3, 0.3, 0.33),
                                                    _slice=None)))

    return html

  def charter_parsing_results_to_html(self, charter: CharterDocument):

    txt_html = self.to_color_text(charter.org['tokens'], charter.org['attention_vector'], _range=[0, 1])

    html = ''
    html += f'<div style="background:#eeeeff; padding:0.5em"> recognized NE(s): ' \
            f'<br><br> ' \
            f'org type:<h3 style="margin:0">{charter.org["type_name"]} </h3>' \
            f'org full name:<h2 style="margin:0">  {charter.org["name"]}</h2> ' \
            f'<br>quote: <div style="font-size:90%; background:white">{txt_html}</div> </div>'

    # html+=txt_html
    html += self.render_constraint_values_2(charter)

    return html

  def _render_sentence(self, sentence: ConstraintsSearchResult):
    # TODO: remove
    # ===========
    print(WARN + f'{self._render_sentence} is deprecated, use {self.constraints_to_html}')
    # ===========

    html = ""
    constraints: List[ValueConstraint] = sentence.constraints

    html += "<br>"
    for probable_v in constraints:
      html += self.value_to_html(probable_v.value)

    if len(constraints) == 0:
      return html

    html += '<div style="border-bottom:1px solid #ccc; margin-top:1em"></div>'

    search_result: PatternSearchResult = sentence.subdoc

    subj_type = search_result.subject_mapping["subj"]
    if subj_type in known_subjects_dict:
      subj_type = known_subjects_dict[subj_type]

    html += as_headline_4(f'{subj_type} <sup>confidence={search_result.subject_mapping["confidence"]:.3f}')

    attention_vectors = self.map_attention_vectors_to_colors(search_result)

    html += as_c_quote(to_multicolor_text(search_result.tokens, attention_vectors,
                                          v_color_map,
                                          min_color=(0.3, 0.3, 0.33),
                                          _slice=None))

    return html

  def map_attention_vectors_to_colors(self, search_result):
    attention_vectors = {
      search_result.attention_vector_name: search_result.get_attention(),
    }
    for subj in known_subjects:
      attention_vectors[AV_PREFIX + f'x_{subj}'] = search_result.get_attention(AV_PREFIX + f'x_{subj}')
      attention_vectors[AV_SOFT + AV_PREFIX + f'x_{subj}'] = search_result.get_attention(
        AV_SOFT + AV_PREFIX + f'x_{subj}')
    return attention_vectors

  def sign_to_text(self, sign: int):
    if sign < 0: return " &lt; "
    if sign > 0: return " &gt; "
    return ' = '

  def probable_value_to_html(self, pv):
    vc = pv.value
    color = '#333333'
    if vc.sign > 0:
      color = '#993300'
    elif vc.sign < 0:
      color = '#009933'

    return f'<b style="color:{color}">{self.sign_to_text(vc.sign)} {vc.currency} {vc.value:20,.2f}' \
           f'<sup>confidence={pv.confidence:20,.2f}</sup></b> '


''' AZ:- 🌈 -----🌈 ------🌈 --------------------------END-Rendering COLORS--------'''


def mixclr(color_map, dictionary, min_color=None, _slice=None):
  reds = None
  greens = None
  blues = None

  fallback = (0.5, 0.5, 0.5)

  for c in dictionary:
    vector = np.array(dictionary[c])
    if _slice is not None:
      vector = vector[_slice]

    if reds is None:
      reds = np.zeros(len(vector))
    if greens is None:
      greens = np.zeros(len(vector))
    if blues is None:
      blues = np.zeros(len(vector))

    vector_color = fallback
    if c in color_map:
      vector_color = color_map[c]

    reds += vector * vector_color[0]
    greens += vector * vector_color[1]
    blues += vector * vector_color[2]

  if min_color is not None:
    reds += min_color[0]
    greens += min_color[1]
    blues += min_color[2]

  def cut_(x):
    up = [min(i, 1) for i in x]
    down = [max(i, 0) for i in up]
    return down

  return np.array([cut_(reds), cut_(greens), cut_(blues)]).T


def to_multicolor_text(tokens, vectors, colormap, min_color=None, _slice=None) -> str:
  if _slice is not None:
    tokens = tokens[_slice]

  colors = mixclr(colormap, vectors, min_color=min_color, _slice=_slice)
  html = ''
  for i in range(len(tokens)):
    c = colors[i]
    r = int(255 * c[0])
    g = int(255 * c[1])
    b = int(255 * c[2])
    if tokens[i] == '\n':
      html += '<br>'
    html += f'<span style="background:rgb({r},{g},{b})">{tokens[i]} </span>'
  return html


''' AZ:- 🌈 -----🌈 ------🌈 --------------------------END-Rendering COLORS--------'''


def _as_smaller(txt):
  return f'<div font-size:12px">{txt}</div>'


def as_c_quote(txt):
  return f'<div style="margin-top:0.2em; margin-left:2em; font-size:14px">"...{txt} ..."</div>'


class BasicProtocolRenderer(HtmlRenderer):

  def protocol_values_to_html(self, _doc: ProtocolDocument) -> str:

    values = sorted(_doc.values, key=lambda item: -item.value.value - item.confidence)

    if values is None or len(values) == 0:
      return as_warning('сумма не найдена')
      return

    cc = 0
    html = ''
    for vl in values:

      h = as_headline_3(self.probable_value_to_html(vl))
      h += as_offset(self.to_color_text(vl.value.context.tokens, vl.value.context.attention, colormap='jet'))
      if cc > 0:
        h = as_offset(as_smaller(h))

      html += h

      cc += 1
    return html
