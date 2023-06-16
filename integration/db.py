import os
import urllib.parse
import warnings

from bson import ObjectId
from pymongo import MongoClient

import gpn_config

_db_client = None




# mongod --config /usr/local/etc/mongod.conf
def get_mongodb_connection():
  global _db_client
  db_name = gpn_config.config.get('GPN_DB_NAME', 'gpn')
  if _db_client is None:
    try:
      host = gpn_config.config.get('GPN_DB_HOST', 'localhost')
      port = gpn_config.config.get('GPN_DB_PORT', 27017)
      print(f"DB HOST IS: {host}")
      user = gpn_config.secret('GPN_DB_USER', None)
      password = gpn_config.secret('GPN_DB_PASSWORD', None)
      mongo_tls = gpn_config.secret('GPN_USE_MONGO_TLS', False)
      ca_file = gpn_config.secret('GPN_DB_TLS_CA')
      cert_file = gpn_config.secret('GPN_DB_TLS_KEY')
      if mongo_tls:
        tls_opts = f'?tls=true&tlsCAFile={ca_file}&tlsCertificateKeyFile={cert_file}'
      else:
        tls_opts = ''
      if user is not None and password is not None:
        user = urllib.parse.quote_plus(user)
        password = urllib.parse.quote_plus(password)
        _db_client = MongoClient(f'mongodb://{user}:{password}@{host}:{port}/{tls_opts}')
      else:
        _db_client = MongoClient(f'mongodb://{host}:{port}/{tls_opts}')
      _db_client.server_info()

    except Exception as err:
      _db_client = None
      msg = f'cannot connect Mongo {err}'
      warnings.warn(msg)
      return None

  return _db_client[db_name]

def get_doc_by_id(doc_id: ObjectId):
  db = get_mongodb_connection()
  documents_collection = db['documents']
  return documents_collection.find_one({'_id': doc_id})


def _get_local_mongodb_connection():
  try:
    _db_client = MongoClient(f'mongodb://localhost:27017/')
    _db_client.server_info()
    return _db_client['gpn']
  except Exception as err:
    msg = f'{err}'
    warnings.warn(msg)
  return None
