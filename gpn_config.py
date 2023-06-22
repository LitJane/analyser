
import inspect
import os
from pathlib import Path

import yaml


import logging

logger = logging.getLogger('gpn_cfg')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"

formatter = logging.Formatter(_FORMAT)
ch.setFormatter(formatter)
logger.setLevel(logging.DEBUG)

logger.addHandler(ch)



def in_unit_test():
  current_stack = inspect.stack()

  for stack_frame in current_stack:
    if stack_frame and stack_frame[4]:
      for program_line in stack_frame[4]:  # This element of the stack frame contains
        if "unittest" in program_line or "pytest" in program_line:  # some contextual program lines

          return True
  return False


in_test_mode = in_unit_test()

path_to_config = os.environ.get('GPN_CONFIG_PATH')
if path_to_config is None:
  path_to_config = Path(__file__).parent / 'config.yml'
  if in_test_mode:
    path_to_config = Path(__file__).parent / 'config-test.yml'
else:
  logger.info('config file is overridden by env var GPN_CONFIG_PATH')
logger.info('config file')
logger.info(path_to_config)

__config = yaml.safe_load(open(str(path_to_config)))


def configured(key, default_val=None):
  val = __config.get(key, default_val)
  if val is None:
    msg = f'⚠️ {key}: config variable is not set, refer {path_to_config}'
    logger.warning(msg)
  return val


def secret(key, default_val=None):
  val = os.environ.get(key, default_val)
  if val is None:
    msg = f'⚠️ {key}: environment variable is not set'
    logger.warning(msg)
  return val


print(path_to_config)
for x in __config:
  print(x, '\t -- \t', f'[{__config[x]}]')

if __name__ == '__main__':
  if in_test_mode:
    pass
    logger.warning("CONFIG: IN UNIT TEST MODE")
