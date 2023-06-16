import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances

import gpn_config
from analyser.hyperparams import models_path, HyperParameters, work_dir
from analyser.legal_docs import tokenize_doc_into_sentences_map, LegalDocument
from analyser.log import logger
from analyser.ml_tools import SemanticTag

__t_cache_dir = gpn_config.configured('TRANSFORMERS_CACHE')
if __t_cache_dir is None:
  __t_cache_dir  = str(Path(work_dir) / 'tf_hub_cache')
os.environ['TRANSFORMERS_CACHE'] = __t_cache_dir


def estimate_distance_threshold(patterns_embeddings):
  distance_matrix = pairwise_distances(patterns_embeddings, patterns_embeddings, metric='cosine', n_jobs=1)

  dshape = distance_matrix.shape
  distance_matrix_meaningful = []
  for i in range(dshape[0]):
    for j in range(i):
      distance_matrix_meaningful.append(distance_matrix[i][j])
  distance_matrix_meaningful = np.array(distance_matrix_meaningful)
  len(distance_matrix_meaningful)

  # mean distance plus/minus tandart deviation .. estimating the max distance from clusters...
  threshold = distance_matrix_meaningful.mean() - distance_matrix_meaningful.std()

  return threshold


class InsidesFinder():
  def __init__(self):
    self.centroids = np.load(str(Path(models_path) / "insides_patterns.npy"))
    self.n_clusters = self.centroids.shape[0]
    logger.info(f'InsidesFinder: centroids.shape {self.centroids.shape}')
    print(f'InsidesFinder: centroids.shape {self.centroids.shape}')

    self.sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    self.sentence_model.max_seq_length = 512

    self.distance_threshold = estimate_distance_threshold(self.centroids)
    print(f'InsidesFinder distance_threshold {self.distance_threshold}')
    logger.debug(f'InsidesFinder distance_threshold {self.distance_threshold}')

  def embedd_sentences(self, strings: []):
    return self.sentence_model.encode(strings)

  def find_insides(self, sample_doc: LegalDocument):

    sentence_map = tokenize_doc_into_sentences_map(sample_doc.tokens_map.get_full_text(),
                                                   HyperParameters.mean_sentense_pattern_len)

    if not hasattr(sample_doc, 'sentences_embeddings_bert'):
      setattr(sample_doc, 'sentences_embeddings_bert', None)

    if sample_doc.sentences_embeddings_bert is None:
      sample_doc.sentences_embeddings_bert = self.embedd_sentences(sentence_map.tokens)

    distance_matrix = pairwise_distances(self.centroids,
                                         sample_doc.sentences_embeddings_bert,
                                         metric='cosine',
                                         n_jobs=1)
    distance_matrix = (distance_matrix * -1) + 1.0

    ########

    sim_max = self.distance_threshold

    for k in range(self.n_clusters):
      av = distance_matrix[k]  # relu(v, threshold) ## attention vector
      ii = av.argmax()

      if av[ii] > sim_max:
        sim_max = av[ii]

        logger.info(f"{k}=cluster \t {av[ii]}=similarity, \n {sentence_map.tokens[ii]} ")
        _span = sentence_map.remap_span((ii, ii + 1), sample_doc.tokens_map)
        tag = SemanticTag('insideInformation', 'Unknown', span=_span, confidence=np.float(av[ii]))
        sample_doc.attributes_tree.insideInformation = tag
