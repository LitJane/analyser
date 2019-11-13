#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from contract_patterns import ContractPatternFactory
from fuzzy_matcher import AttentionVectors, prepare_patters_for_embedding_2
from patterns import FuzzyPattern
from text_tools import Tokens, hot_punkt


def make_punkt_mask(tokens: Tokens) -> np.ndarray:
  return 1.0 - 0.999 * hot_punkt(tokens)


class AbstractElmoBasedModel:
  __elmo_shared = None

  def __init__(self, module_url: str = 'https://storage.googleapis.com/az-nlp/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz'):
    self.module_url = module_url
    self.embedding_session = None
    self.elmo = self.__elmo_shared

  def _build_graph(self) -> None:
    # BUILD IT -----------------------------------------------------------------
    if self.elmo is None:
      self.elmo = hub.Module(self.module_url, trainable=False)
      self.__elmo_shared = self.elmo

  def make_embedding_session_and_graph(self):
    embedding_graph = tf.Graph()

    with embedding_graph.as_default():
      self._build_graph()

      init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])

      self.embedding_session = tf.Session(graph=embedding_graph)
      self.embedding_session.run(init_op)

    embedding_graph.finalize()

  @staticmethod
  def _embed(elmo, text_input_p, lengths_p):
    # 1. text embedding
    return elmo(
      inputs={
        "tokens": text_input_p,
        "sequence_len": lengths_p
      },
      signature="tokens",
      as_dict=True)["elmo"]


class PatternSearchModel(AbstractElmoBasedModel):

  def __init__(self,
               module_url='https://storage.googleapis.com/az-nlp/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz'):

    AbstractElmoBasedModel.__init__(self, module_url)

    self._text_range = None
    self._patterns_range = None

    self.cosine_similarities = None

    self.make_embedding_session_and_graph()

  def _center(self, x):
    return tf.reduce_mean(x, axis=0)

  def _pad_tensor(self, tensor, padding, el, name):
    _mask_padding = tf.tile(el, padding, name='pad_' + name)
    return tf.concat([tensor, _mask_padding], axis=0, name='pad_tensor_' + name)

  def _build_graph(self) -> None:
    super()._build_graph()

    # inputs:--------------------------------------------------------------------
    self.text_input = tf.placeholder(dtype=tf.string, name="text_input")
    self.text_lengths = tf.placeholder(dtype=tf.int32, name='text_lengths')

    self.pattern_input = tf.placeholder(dtype=tf.string, name='pattern_input')
    self.pattern_lengths = tf.placeholder(dtype=tf.int32, name='pattern_lengths')
    self.pattern_slices = tf.placeholder(dtype=tf.int32, name='pattern_slices', shape=[None, 2])
    # ------------------------------------------------------------------- /inputs

    patterns_max_len = tf.math.reduce_max(self.pattern_lengths, keepdims=True, name='patterns_max_len')

    number_of_patterns = tf.shape(self.pattern_input)[0]

    text_input_padded = [
      self._pad_tensor(self.text_input[0], patterns_max_len, tf.constant(['\n']), name='text_input_ext')]

    # 1. text embedding---------------
    # TODO: try to deal with segmented text (this is about trailing index - [0] )
    self._text_embedding = self._embed(self.elmo,
                                       text_input_padded,
                                       [self.text_lengths[0] + patterns_max_len[0]])[0]

    # 2. patterns embedding
    _patterns_embeddings = self._embed(self.elmo, self.pattern_input, self.pattern_lengths)

    # ranges for looping
    self._text_range = tf.range(self.text_lengths[0], dtype=tf.int32, name='text_input_range')
    self._patterns_range = tf.range(number_of_patterns, dtype=tf.int32, name='patterns_range')
    self.cosine_similarities = tf.map_fn(
      lambda i: self.for_every_pattern((self.pattern_lengths[i], self.pattern_slices[i], _patterns_embeddings[i]),
                                       self._text_embedding,
                                       self._text_range), self._patterns_range, dtype=tf.float32,
      name='cosine_similarities')

  def _normalize(self, x):
    #       _norm = tf.norm(x, keep_dims=True)
    #       return x/ (_norm + 1e-8)
    return tf.nn.l2_normalize(x, 0)  # TODO: try different norm

  def get_vector_similarity(self, a, b):
    a_norm = self._normalize(a)  # normalizing is kinda required if we want cosine return [0..1] range
    b_norm = self._normalize(b)  # DO WE? TODO: try different norm
    return 1.0 - tf.losses.cosine_distance(a_norm, b_norm, axis=0)  # TODO: how on Earth Cosine could be > 1????

  def get_matrix_vector_similarity(self, matrix, vector):
    m_center = self._center(matrix)
    return self.get_vector_similarity(vector, m_center)

  def for_every_pattern(self, pattern_info, _text_embedding, text_range):
    pattern_slice = pattern_info[1]

    _patterns_embeddings = pattern_info[2]
    pattern_emb_sliced = _patterns_embeddings[pattern_slice[0]: pattern_slice[1]]

    return self._convolve(text_range, _text_embedding, pattern_emb_sliced, name='p_match')

  def _convolve(self, text_range, _text_embedding, pattern_emb_sliced, name=''):

    window_size = tf.shape(pattern_emb_sliced)[0]

    p_center = self._center(pattern_emb_sliced)

    _blurry = tf.map_fn(
      lambda i: self.get_matrix_vector_similarity(matrix=_text_embedding[i:i + window_size],
                                                  vector=p_center),
      text_range, dtype=tf.float32, name=name + '_sim_wnd')

    _sharp = tf.map_fn(
      lambda i: self.get_matrix_vector_similarity(matrix=_text_embedding[i:i + 1],
                                                  vector=p_center),
      text_range, dtype=tf.float32, name=name + '_sim_w1')

    return tf.math.maximum(_blurry, _sharp, name=name + '_merge')

  # ------

  def _fill_dict(self, text_tokens: Tokens, patterns: List[FuzzyPattern]):
    patterns_tokens, patterns_lengths, pattern_slices, _ = prepare_patters_for_embedding_2(patterns)
    feed_dict = {
      self.text_input: [text_tokens],  # text_input
      self.text_lengths: [len(text_tokens)],  # text_lengths

      self.pattern_input: patterns_tokens,
      self.pattern_lengths: patterns_lengths,
      self.pattern_slices: pattern_slices

    }
    return feed_dict

  def find_patterns(self, text_tokens: Tokens, patterns: List[FuzzyPattern]) -> AttentionVectors:
    for t in text_tokens:
      assert t is not None
      assert len(t) > 0

    runz = [self.cosine_similarities]

    feed_dict = self._fill_dict(text_tokens, patterns)
    attentions = self.embedding_session.run(runz, feed_dict=feed_dict)[0]

    av = AttentionVectors()

    for i in range(len(patterns)):
      pattern = patterns[i]
      av.add(pattern.name, attentions[i])

    return av


class PatternSearchModelExt(PatternSearchModel):

  def __init__(self, module_url: str = 'https://storage.googleapis.com/az-nlp/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz'):
    PatternSearchModel.__init__(self, module_url)

  # ------
  def get_distances_to_center(self, text_tokens: Tokens):
    runz = [self.distances_to_center, self.distances_to_local_center]
    d, dl = self.embedding_session.run(runz, feed_dict={
      self.text_input: [text_tokens],  # text_input
      self.text_lengths: [len(text_tokens)],  # text_lengths

      self.pattern_input: [['a']],
      self.pattern_lengths: [1],
      self.pattern_slices: [[0, 1]]

    })

    return d, dl

  def _build_graph(self) -> None:
    super()._build_graph()

    def improve_dist(attention_vector, pattern_len):
      """
        finding closest point (aka 'best point')
      """
      max_i = tf.math.argmax(attention_vector, output_type=tf.dtypes.int32)
      best_embedding_range = self._text_embedding[max_i:max_i + pattern_len]  # metapattern

      return self._convolve(self._text_range, self._text_embedding, pattern_emb_sliced=best_embedding_range,
                            name='improving')

    def find_best_embeddings():
      return tf.map_fn(
        lambda pattern_i: improve_dist(self.cosine_similarities[pattern_i], self.pattern_lengths[pattern_i]),
        self._patterns_range, dtype=tf.float32, name="find_best_embeddings")

    self.cosine_similarities_improved = find_best_embeddings()

    unpadded_text_embedding_ = self._embed(self.elmo,
                                           self.text_input,
                                           [self.text_lengths[0]])[0]

    text_center = self._center(unpadded_text_embedding_)

    self.distances_to_center = tf.map_fn(
      lambda i: self.get_matrix_vector_similarity(unpadded_text_embedding_[i:i + 1],
                                                  text_center),
      self._text_range, dtype=tf.float32)
    word_text_center_i = tf.math.argmax(self.distances_to_center, output_type=tf.dtypes.int32)
    word_text_center = unpadded_text_embedding_[word_text_center_i]
    self.distances_to_local_center = tf.map_fn(
      lambda i: self.get_matrix_vector_similarity(unpadded_text_embedding_[i:i + 1],
                                                  word_text_center), self._text_range, dtype=tf.float32)

  def find_patterns_and_improved(self, text_tokens: Tokens, patterns: List[FuzzyPattern]) -> AttentionVectors:
    runz = [self.cosine_similarities, self.cosine_similarities_improved]

    feed_dict = self._fill_dict(text_tokens, patterns)
    attentions, improved_attentions = self.embedding_session.run(runz, feed_dict=feed_dict)

    av = AttentionVectors()

    for i in range(len(patterns)):
      pattern = patterns[i]
      av.add(pattern.name, attentions[i], improved_attentions[i])

    return av


from tensorflow.python.summary.writer.writer import FileWriter


class PatternSearchModelNoEmb:

  def __init__(self):
    self._text_range = None
    self._patterns_range = None

    self.cosine_similarities = None

    self._text_embedding = None
    self._patterns_embeddings = None
    self.pattern_slices = None

    embedding_graph = tf.Graph()

    with embedding_graph.as_default():
      self._build_graph()

      init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])

      self.embedding_session = tf.Session(graph=embedding_graph)
      self.embedding_session.run(init_op)

    embedding_graph.finalize()
    FileWriter('logs/train', graph=embedding_graph).close()

  def _center(self, x):
    return tf.reduce_mean(x, axis=0)

  def _build_graph(self) -> None:
    # inputs:--------------------------------------------------------------------
    with tf.name_scope('text_embedding') as scope:
      self._text_embedding = tf.placeholder(dtype=tf.float32, name='text_embedding', shape=[None, 1024])

    with tf.name_scope('patterns') as scope:
      self._patterns_embeddings = tf.placeholder(dtype=tf.float32, name='patterns_embeddings', shape=[None, None, 1024])
      self.pattern_slices = tf.placeholder(dtype=tf.int32, name='pattern_slices', shape=[None, 2])
      # ------------------------------------------------------------------- /inputs

    with tf.name_scope('ranges') as scope:
      # ranges for looping
      number_of_patterns = tf.shape(self._patterns_embeddings)[-2]
      text_len = tf.shape(self._text_embedding)[-1]

      self._text_range = tf.range(text_len, dtype=tf.int32, name='text_input_range')
      self._patterns_range = tf.range(number_of_patterns, dtype=tf.int32, name='patterns_range')

    self.cosine_similarities = tf.map_fn(
      lambda i: self.for_every_pattern((self.pattern_slices[i], self._patterns_embeddings[i]),
                                       self._text_embedding,
                                       self._text_range), self._patterns_range, dtype=tf.float32,
      name='cosine_similarities')

  def _normalize(self, x):
    #       _norm = tf.norm(x, keep_dims=True)
    #       return x/ (_norm + 1e-8)
    return tf.nn.l2_normalize(x, 0, name='l2_norm')  # TODO: try different norm

  def get_vector_similarity(self, a, b):
    with tf.name_scope('similarity') as scope:
      a_norm = self._normalize(a)  # normalizing is kinda required if we want cosine return [0..1] range
      b_norm = self._normalize(b)  # DO WE? TODO: try different norm
      return 1.0 - tf.losses.cosine_distance(a_norm, b_norm, axis=0)  # TODO: how on Earth Cosine could be > 1????

  def get_matrix_vector_similarity(self, matrix, vector):
    with tf.name_scope('get_matrix_vector_similarity') as scope:
      m_center = self._center(matrix)
      return self.get_vector_similarity(vector, m_center)

  def for_every_pattern(self, pattern_info: tuple, _text_embedding, text_range):
    pattern_slice = pattern_info[0]
    _patterns_embeddings = pattern_info[1]
    pattern_emb_sliced = _patterns_embeddings[pattern_slice[0]: pattern_slice[1]]

    return self._convolve(text_range, _text_embedding, pattern_emb_sliced, name='p_match')

  def _convolve(self, text_range, _text_embedding, pattern_emb_sliced, name=''):
    window_size = tf.shape(pattern_emb_sliced)[0]
    p_center = self._center(pattern_emb_sliced)

    _blurry = tf.map_fn(
      lambda i: self.get_matrix_vector_similarity(matrix=_text_embedding[i:i + window_size], vector=p_center),
      text_range, dtype=tf.float32, name=name + '_sim_wnd')

    _sharp = tf.map_fn(
      lambda i: self.get_matrix_vector_similarity(matrix=_text_embedding[i:i + 1], vector=p_center),
      text_range, dtype=tf.float32, name=name + '_sim_w1')

    return tf.math.maximum(_blurry, _sharp, name=name + '_merge')

  # -------------------

  def find_patterns(self, text_embedding, patterns_embeddings, pattern_slices) -> AttentionVectors:
    runz = [self.cosine_similarities]

    feed_dict = {
      self._text_embedding: text_embedding,  # text_input
      self._patterns_embeddings: patterns_embeddings,  # text_lengths
      self.pattern_slices: pattern_slices
    }

    attentions = self.embedding_session.run(runz, feed_dict=feed_dict)[0]
    return attentions


class PatternSearchModelDiff:

  def __init__(self):
    self._text_range = None
    self._patterns_range = None

    self._text_embedding = None
    self._patterns_embeddings = None
    self.pattern_slices = None
    self._similarities = None
    self.similarities_norm = None

    self.embedding_session = None

    embedding_graph = tf.Graph()

    with embedding_graph.as_default():
      self._build_graph()

      init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])

      self.embedding_session = tf.Session(graph=embedding_graph)
      self.embedding_session.run(init_op)

    embedding_graph.finalize()
    FileWriter('logs/train', graph=embedding_graph).close()

  def _build_graph(self) -> None:
    # inputs:--------------------------------------------------------------------
    with tf.name_scope('text_embedding') as scope:
      self._text_embedding = tf.placeholder(dtype=tf.float32, name='text_embedding', shape=[None, 1024])
      self._text_embedding = tf.nn.l2_normalize(self._text_embedding, name='text_embedding_norm')

      paddings = tf.constant([[0, 20, ], [0, 0]])
      self._text_embedding_p = tf.pad(self._text_embedding, paddings, "SYMMETRIC")

    with tf.name_scope('patterns') as scope:
      self._patterns_embeddings = tf.placeholder(dtype=tf.float32, name='patterns_embeddings', shape=[None, None, 1024])
      self._patterns_embeddings = tf.nn.l2_normalize(self._patterns_embeddings, name='patterns_embeddings_norm')
      self.pattern_slices = tf.placeholder(dtype=tf.int32, name='pattern_slices', shape=[None, 2])
      # ------------------------------------------------------------------- /inputs

    with tf.name_scope('ranges') as scope:
      # ranges for looping
      number_of_patterns = tf.shape(self.pattern_slices)[-2]
      text_len = tf.shape(self._text_embedding)[-2]

      self._text_range = tf.range(text_len, dtype=tf.int32, name='text_input_range')
      self._patterns_range = tf.range(number_of_patterns, dtype=tf.int32, name='patterns_range')

    def cc(i):
      return self.for_every_pattern((self.pattern_slices[i], self._patterns_embeddings[i]))

    self.similarities_norm = tf.map_fn(cc, self._patterns_range, dtype=tf.float32, name='distances')

  def for_every_pattern(self, pattern_info: tuple):
    _slice = pattern_info[0]
    _patterns_embeddings = pattern_info[1]
    pattern_emb_sliced = _patterns_embeddings[_slice[0]: _slice[1]]

    return self._convolve(pattern_emb_sliced, name='p_match')

  def _convolve(self, pattern_emb_sliced, name=''):
    window_size = tf.shape(pattern_emb_sliced)[-2]

    def cc(i):
      return self.get_cosine_similarity(m1=self._text_embedding_p[i:i + window_size], m2=pattern_emb_sliced)

    _blurry = tf.map_fn(cc, self._text_range, dtype=tf.float32, name=name + '_sim_wnd')

    return _blurry

  def get_cosine_similarity(self, m1, m2):
    with tf.name_scope('cosine_similarity') as scope:
      uv = tf.math.reduce_mean(tf.math.multiply(m1, m2))
      uu = tf.math.reduce_mean(tf.math.square(m1))
      vv = tf.math.reduce_mean(tf.math.square(m2))
      dist = uv / tf.math.sqrt(tf.math.multiply(uu, vv))
      return dist

  def find_patterns(self, text_embedding, patterns_embeddings, pattern_slices):
    runz = [self.similarities_norm]

    feed_dict = {
      self._text_embedding: text_embedding,  # text_input
      self._patterns_embeddings: patterns_embeddings,  # text_lengths
      self.pattern_slices: pattern_slices
    }

    attentions = self.embedding_session.run(runz, feed_dict=feed_dict)[0]
    return attentions


if __name__ == '__main__':
  from text_tools import tokenize_text
  import re
  from text_normalize import normalize_text, replacements_regex

  search_for = "Изящество стиля"

  from text_tools import tokenize_text
  import re
  from text_normalize import normalize_text, replacements_regex

  _sample_text = """
  Берлинская дада-ярмарка и Международная выставка сюрреализма в париж­ской галерее «Изящные искусства» в 1938 году стали высшими точками развития двух движений и подвели им итог. На «Сюрреалис­тической улице», за манекенами, выстроившимися в проходе в главный зал, располагались плакаты, приглашения, объявления и фотографии, отсылающие к ранним этапам сюрреализма. В главном зале, за оформление которого отвечал Марсель Дюшан 

  , а за освеще­ние — Ман Рэй 

  , картины 1920-х годов висели рядом с более ранними работами, что подчеркивало развити
  е сюрреалистического «интернаци­она­ла». Зародившись как литературное течение, к концу 1930-х годов сюрреализм уже около 15 лет господствовал в художественном авангарде Парижа. Прежде чем пойти на спад с началом
  Второй мировой войны, он стал частью светской культуры Парижа и даже до некотоРой
  степени присягнул высокой моде, подобно тому как русский авангард — пусть совсем иначе — присягну 
  в свое время революции. Изящество стиля, свойственное сюррелизму, способствоало этому 

  ­ сближению, которое, в свою очередь, упрочило положение многих представителей направления в обществе. 

  Однако поначалу для литераторов и художников-бунтарей, ничуть не стремившихся к социаль­ному успеху, 
  была куда более естественной связь с дадаизмом"""

  _regex_addon = [
    (re.compile(r'[­]'), '-'),
  ]
  TOKENS = tokenize_text(normalize_text(_sample_text, replacements_regex + _regex_addon))

  patterns = ContractPatternFactory().patterns
  prepare_patters_for_embedding_2(patterns)
  PM = PatternSearchModel()
  av = PM.find_patterns(text_tokens=TOKENS, patterns=patterns)
  # print(av)
