import os
import warnings

from analyser.log import logger


def _env_var(vname, default_val=None):
  warnings.warn("use gpn_config instead", DeprecationWarning)
  if vname not in os.environ:
    msg = f'SYSTEM : define {vname} environment variable! defaulting to {default_val}'
    logger.warning(msg)
    return default_val
  else:
    return os.environ[vname]
