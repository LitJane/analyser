import os
import warnings

from pymongo import MongoClient


def __check_var(vname):
  if vname not in os.environ:
    msg = f'MongoDB could not be connected, define {vname} environment variable'
    warnings.warn(msg)
    return False
  else:
    return True

# mongod --config /usr/local/etc/mongod.conf

def get_mongodb_connection():
  if __check_var('GPN_DB_HOST') and __check_var('GPN_DB_PORT') and __check_var('GPN_DB_NAME'):
    client = MongoClient(f'mongodb://{os.environ["GPN_DB_HOST"]}:{os.environ["GPN_DB_PORT"]}/')
    return client[os.environ["GPN_DB_NAME"]]
  return None


if __name__ == '__main__':
  get_mongodb_connection()
