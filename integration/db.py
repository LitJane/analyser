import os
import urllib.parse
import warnings

from pymongo import MongoClient

_db_client = None


def _env_var(vname, default_val=None):
  if vname not in os.environ:
    msg = f'MongoDB : define {vname} environment variable! defaulting to {default_val}'
    warnings.warn(msg)
    return default_val
  else:
    return os.environ[vname]


# mongod --config /usr/local/etc/mongod.conf
def get_mongodb_connection():
  global _db_client
  db_name = _env_var('GPN_DB_NAME', 'gpn')
  if _db_client is None:
    try:
      host = _env_var('GPN_DB_HOST', 'localhost')
      port = _env_var('GPN_DB_PORT', 27017)
      print(f"DB HOST IS: {host}")
      user = _env_var('GPN_DB_USER', None)
      password = _env_var('GPN_DB_PASSWORD', None)
      if user is not None and password is not None:
        user = urllib.parse.quote_plus(user)
        password = urllib.parse.quote_plus(password)
        _db_client = MongoClient(f'mongodb://{user}:{password}@{host}:{port}/')
      else:
        _db_client = MongoClient(f'mongodb://{host}:{port}/')
      _db_client.server_info()

    except Exception as err:
      _db_client = None
      msg = f'cannot connect Mongo {err}'
      warnings.warn(msg)
      return None

  return _db_client[db_name]


def _get_local_mongodb_connection():
  try:
    _db_client = MongoClient(f'mongodb://localhost:27017/')
    _db_client.server_info()
    return _db_client['gpn']
  except Exception as err:
    msg = f'{err}'
    warnings.warn(msg)
  return None
