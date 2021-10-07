from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances

from analyser.hyperparams import models_path, HyperParameters
from analyser.legal_docs import tokenize_doc_into_sentences_map, LegalDocument
from analyser.log import logger
from analyser.ml_tools import SemanticTag
from tf_support.embedder_elmo import ElmoEmbedder


class InsidesFinder():
  def __init__(self):
    self.centroids = np.load(str(Path(models_path) / "insides_patterns.npy"))
    self.n_clusters = self.centroids.shape[0]
    logger.info(f'centroids.shape {self.centroids.shape}')

  def find_insides(self, sample_doc: LegalDocument):
    if not hasattr(sample_doc, 'sentence_map'):
      #todo: remove this hack
      setattr(sample_doc, 'sentence_map', None)

    if sample_doc.sentence_map is None:
      sample_doc.sentence_map = tokenize_doc_into_sentences_map(sample_doc.tokens_map.get_full_text(),
                                                                HyperParameters.mean_sentense_pattern_len)

    if not hasattr(sample_doc, 'sentences_embeddings'):
      setattr(sample_doc, 'sentences_embeddings', None)
    if sample_doc.sentences_embeddings is None:
      embedder = ElmoEmbedder.get_instance()
      sample_doc.sentences_embeddings = embedder.embedd_strings(sample_doc.sentence_map.tokens)

    X = sample_doc.sentences_embeddings
    distance_matrix = pairwise_distances(X, self.centroids, metric='cosine', n_jobs=1)
    # distance_matrix = relu ( ((distance_matrix * -1)+1) , _mx-0.01)

    distance_matrix = (distance_matrix * -1) + 1.0
    distance_matrix = distance_matrix.T

    ########

    # TODO: parametrize
    threshold = 0.85  # 0.9 *  distance_matrix.max()

    sim_max = threshold

    for k in range(self.n_clusters):
      av = distance_matrix[k]  # relu(v, threshold) ## attention vector
      ii = av.argmax()

      if av[ii] > sim_max:
        logger.info(f"{k}=cluster \t {av[ii]}=similarity, \n {sample_doc.sentence_map.tokens[ii]} ")
        # char_span = sample_doc.sentence_map.map[ii]

        _span = sample_doc.sentence_map.remap_span((ii, ii + 1), sample_doc.tokens_map)
        # logger.info(f"span (chars):  {char_span}, {_span}")
        tag = SemanticTag('insideInformation', 'Unknown', span=_span, confidence=np.float(av[ii]))

        sim_max = av[ii]

        sample_doc.attributes_tree.insideInformation = tag
        # setattr(sample_doc.attributes_tree, "insideInformation", tag)

    # print(sim_max, i_max)

# if __name__ == '__main__':
#   TEST_DOC_ID = '61408a6e11c893efc81ddcb8'
#
#   sample_id = ObjectId(TEST_DOC_ID)
#
#   sample_db_doc = get_doc_by_id(sample_id)
#   sample_j_doc = DbJsonDoc(sample_db_doc)
#   sample_doc = sample_j_doc.asLegalDoc()
#
#   print(type(sample_doc))
#   iff = InsidesFinder()
#   iff.find_insides(sample_doc)
#
#   print(sample_doc.attributes_tree.insideInformation)
