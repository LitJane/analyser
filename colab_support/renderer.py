import html as escaper
from typing import List

import numpy as np

from analyser.legal_docs import LegalDocument
from analyser.ml_tools import ProbableValue
from analyser.parsing import known_subjects
from analyser.patterns import AV_PREFIX, AV_SOFT
from analyser.structures import ContractSubject
from analyser.structures import OrgStructuralLevel
from analyser.transaction_values import ValueConstraint

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

  def _to_color_text(self, _tokens, weights, mpl, colormap='coolwarm', _range=None, separator=' '):
    tokens = [escaper.escape(t) for t in _tokens]

    if len(tokens) == 0:
      return " - empty -"

    if len(weights) != len(tokens):
      raise ValueError("number of weights differs weights={} tokens={}".format(len(weights), len(tokens)))

    #   if()
    vmin = weights.min() - 0.00001
    vmax = weights.max() + 0.00001

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


def print_headers(contract: LegalDocument):
  for p in contract.paragraphs:
    print('\t --> 📂', contract.substr(p.header))
