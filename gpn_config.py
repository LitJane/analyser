import os
import sys
from pathlib import Path

import yaml
from analyser.log import logger

in_test_mode = 'unittest' in sys.modules.keys()

path_to_config = os.environ.get('GPN_CONFIG_PATH')
if path_to_config is None:
  path_to_config = Path(__file__).parent / 'config.yml'
  if in_test_mode:
    path_to_config = Path(__file__).parent / 'config-test.yml'
else:
  logger.info('config file is overriden by env var GPN_CONFIG_PATH')
logger.info('config file')
logger.info(path_to_config)


config = yaml.safe_load(open(str(path_to_config)))


def secret(key, default_val=None):
  return os.environ.get(key, default_val)

if __name__ == '__main__':
  print(config['MLFLOW_URL'])
