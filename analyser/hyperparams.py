import os
import pathlib
import warnings
from pathlib import Path
import gpn_config

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

from analyser.log import logger

__location__path = Path(__location__)
work_dir: Path or None = None
if gpn_config.configured('GPN_WORK_DIR'):
  work_dir = Path(gpn_config.configured('GPN_WORK_DIR'))
else:
  work_dir = __location__path.parent.parent / 'work'
  warnings.warn('please set GPN_WORK_DIR environment variable')

datasets_dir: Path = work_dir / 'datasets'
reports_dir: Path = Path(__file__).parent / 'training_reports'
notebooks_dir: Path = Path(__file__).parent / 'trainsets'
models_path = str(__location__path / 'vocab')

print(f'USING WORKDIR: [{work_dir}]\n configure GPN_WORK_DIR to override')

pathlib.Path(datasets_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(reports_dir).mkdir(parents=True, exist_ok=True)

logger.info('⚙️ work_dir      [%s]', work_dir)
logger.info('⚙️ models_path   [%s]', models_path)
logger.info('⚙️ reports_dir   [%s]', reports_dir)
logger.info('⚙️ datasets_dir  [%s]', datasets_dir)
logger.info('⚙️ notebooks_dir [%s]', notebooks_dir)


__t_cache_dir = gpn_config.configured('TFHUB_CACHE_DIR')
if __t_cache_dir is None:
  __t_cache_dir = work_dir / 'tf_hub_cache'
os.environ['TFHUB_CACHE_DIR'] = str(__t_cache_dir)


__t_cache_dir = gpn_config.configured('TRANSFORMERS_CACHE')
if __t_cache_dir is None:
  __t_cache_dir  = str(Path(work_dir) / 'tf_hub_cache')
os.environ['TRANSFORMERS_CACHE'] = __t_cache_dir
del __t_cache_dir

class HyperParameters:
  mean_sentense_pattern_len = 300

  max_sentenses_to_embedd = 160

  max_doc_size_tokens = 15000

  max_doc_size_chars = max_doc_size_tokens * 5

  protocol_caption_max_size_words = 200

  sentence_max_len = 200
  charter_sentence_max_len = sentence_max_len
  protocol_sentence_max_len = sentence_max_len

  subsidiary_name_match_min_jaro_similarity = 0.9

  confidence_epsilon = 0.001

  parser_headline_attention_vector_denominator = 0.75

  header_topic_min_confidence = 0.7

  org_level_min_confidence = 0.8

  subject_paragraph_attention_blur = 10

  charter_charity_attention_confidence = 0.6
  charter_subject_attention_confidence = 0.66

  obligations_date_pattern_threshold = 0.4
  hdbscan_cluster_proximity = 0.8

  embedding_window = 3500
  max_doc_size_tokens_for_training = 3500


if __name__ == '__main__':
  print(__location__)
