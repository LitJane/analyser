import html as escaper
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from sklearn.metrics import confusion_matrix

from analyser.legal_docs import LegalDocument
from analyser.ml_tools import ProbableValue
from analyser.structures import ContractSubject
from analyser.structures import OrgStructuralLevel
from analyser.text_tools import Tokens
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

known_subjects = [
  ContractSubject.Charity,
  ContractSubject.RealEstate,
  ContractSubject.Lawsuit]

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
  if v is None:
    return "any"
  return f'{v.value:20,.0f} {v.currency} '


class AbstractRenderer:

  def sign_to_text(self, sign: int):
    if sign < 0:
      return " < "
    if sign > 0:
      return " > "
    return ' = '

  def sign_to_html(self, sign: int):
    if sign < 0:
      return " &lt; "
    if sign > 0:
      return " &gt; "
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


colors_by_contract_subject = {
  ContractSubject.RealEstate: (0.2, 0.2, 1),
  ContractSubject.Lawsuit: (0.9, 0, 0.9),
  ContractSubject.Charity: (0.0, 0.9, 0.3),
}


class HtmlRenderer(AbstractRenderer):

  def _to_color_text(self, _tokens, weights, mpl_, colormap='coolwarm', _range=None, separator=' '):
    tokens = [escaper.escape(t) for t in _tokens]

    if len(tokens) == 0:
      return " - empty -"

    if len(weights) != len(tokens):
      raise ValueError("number of weights differs weights={} tokens={}".format(len(weights), len(tokens)))

    vmin = weights.min() - 0.00001
    vmax = weights.max() + 0.00001

    if _range is not None:
      vmin = _range[0]
      vmax = _range[1]

    norm = mpl_.colors.Normalize(vmin=vmin - 0.5, vmax=vmax)
    cmap = mpl_.cm.get_cmap(colormap)

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

  def sign_to_text(self, sign: int):
    if sign < 0:
      return " &lt; "
    if sign > 0:
      return " &gt; "
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

  _colors = mixclr(colormap, vectors, min_color=min_color, _slice=_slice)
  html = ''
  for i, token in enumerate(tokens):
    c = _colors[i]
    r = int(255 * c[0])
    g = int(255 * c[1])
    b = int(255 * c[2])
    if token == '\n':
      html += '<br>'
    html += f'<span style="background:rgb({r},{g},{b})">{token} </span>'
  return html


''' AZ:- 🌈 -----🌈 ------🌈 --------------------------END-Rendering COLORS--------'''


def _as_smaller(txt):
  return f'<div font-size:12px">{txt}</div>'


def as_c_quote(txt):
  return f'<div style="margin-top:0.2em; margin-left:2em; font-size:14px">"...{txt} ..."</div>'


def print_headers(contract: LegalDocument):
  for p in contract.paragraphs:
    print('\t --> 📂', contract.substr(p.header))


def _to_color_text(_tokens, weights, colormap='coolwarm', _range=None, separator=' '):
  tokens = [escaper.escape(t) for t in _tokens]

  if len(tokens) == 0:
    return " - empty -"

  if len(weights) != len(tokens):
    raise ValueError("number of weights differs weights={} tokens={}".format(len(weights), len(tokens)))

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


def to_color_text(tokens, weights, colormap='coolwarm', _range=None, separator=' '):
  return _to_color_text(tokens, weights, colormap=colormap, _range=_range, separator=separator)


def render_spans(spans, subdoc, attention_v, ht='') -> str:
  ht += '<ol>'
  for span in spans:
    _s = slice(span[0], span[1])

    ht += '<li>'
    t = subdoc.tokens_map.tokens[_s]
    att = attention_v[_s]
    ht += to_color_text(t, att, _range=(0, 1.2))
    ht += '<br><hr>'
    ht += '</li>'
  ht += '</ol>'

  return ht


def render_slices(slices, subdoc, attention_v, ht='') -> str:
  ht += '<ol>'
  for _s in slices:
    ht += '<li>'
    t = subdoc.tokens_map.tokens[_s]
    att = attention_v[_s]
    ht += to_color_text(t, att, _range=(0, 1.2))
    ht += '<br><hr>'
    ht += '</li>'
  ht += '</ol>'

  return ht


def rgb_by_string(s):
  h = abs(hash(s))
  r = (h % 256)
  g = int(h / 256.) % 256
  b = 255 - g
  return (r, g, b)


def color_by_string(s, palette=None) -> str:
  h = abs(hash(s))
  if palette is None:
    r = (h % 256)
    g = int(h / 256.) % 256
    b = 255 - g
    return f'#{r:02x}{g:02x}{b:02x}'
  else:
    return palette[h % (len(palette))]


def hex2rgb(c: str):
  return [int(c[1:3], base=16), int(c[3:5], base=16), int(c[5:], base=16)]


def lerp_int(a, b, w):
  return int(a * w + b * (1. - w))


def lerp_float(a, b, w):
  return a * w + b * (1. - w)


def whiter(c: str, weight: float):
  rgb = hex2rgb(c)
  return colors.to_hex([lerp_float(a / 255., 1, weight) for a in rgb])


def render_token_clusters(tokens: Tokens, clusters: [int], pal: [str], weights=None) -> str:
  html = ''
  separator = ' '

  for d, word in enumerate(tokens):

    if word == ' ':
      word = '&nbsp;_ '
    token_color = pal[clusters[d]]
    if weights is not None:
      _w = max(0.5, min(weights[d], 1))
      token_color = whiter(token_color, _w)

    html += f'<span title="{d} {clusters[d]:.2f}" style="background-color:{token_color}">{word}{separator}</span>'

    if tokens[d] == '\n':
      html += "¶<br>"

  return html


def render_doc(doc, semantic_map, default_color='#eeeeee', palette: [str] or None = None):
  _pal = [default_color] + [color_by_string(c, palette) for c in semantic_map.columns]

  clusters: [int] = np.argmax(semantic_map.values, axis=1)
  wieghts = np.max(semantic_map.values, axis=1)

  h = ''
  for p in doc.paragraphs:
    s: slice = p.body.as_slice()
    hs: slice = p.header.as_slice()

    head_html = render_token_clusters(doc.tokens[hs], clusters[hs], _pal, wieghts[hs])
    h += f'<h3>{head_html}</h3>'
    paragraph_html = render_token_clusters(doc.tokens[s], clusters[s], _pal, wieghts[s])
    h += f'<p style="padding:0.5cm 1cm">{paragraph_html}</p>'

  return h


def plot_embedding(matrix, title=None, width=25, height=6):
  mt = matrix.T
  with sns.axes_style("white"):
    fig = plt.figure(figsize=(width, height))
    sns.heatmap(mt, square=False, cmap="RdYlBu", center=0)
    if title:
      plt.title(title)
    plt.show()
    return fig


def plot_cm(y_true, y_pred, figsize=(12, 12)):
  cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
  cm_sum = np.sum(cm, axis=1, keepdims=True)
  cm_perc = cm / cm_sum.astype(float) * 100
  annot = np.empty_like(cm).astype(str)
  nrows, ncols = cm.shape
  for i in range(nrows):
    for j in range(ncols):
      c = cm[i, j]
      p = cm_perc[i, j]
      if i == j:
        s = cm_sum[i]
        annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
      elif c == 0:
        annot[i, j] = ''
      else:
        annot[i, j] = '%.1f%%\n%d' % (p, c)
  cm = pd.DataFrame(cm_perc, index=np.unique(y_true), columns=np.unique(y_true))
  cm.index.name = 'Actual'
  cm.columns.name = 'Predicted'
  fig, ax = plt.subplots(figsize=figsize)
  sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax)
