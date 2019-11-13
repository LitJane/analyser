import tensorflow as tf
from tensorflow.python.summary.writer.writer import FileWriter


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
