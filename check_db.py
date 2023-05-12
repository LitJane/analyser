#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


from pymongo import ASCENDING

from integration.db import get_mongodb_connection

if __name__ == '__main__':
  db = get_mongodb_connection()
  audits_collection = db['audits']

  coll = db["documents"]
  sorting = [('analysis.analyze_timestamp', ASCENDING), ('user.updateDate', ASCENDING)]
  resp = coll.create_index(sorting)
  print("index response:", resp)
