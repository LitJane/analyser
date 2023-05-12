#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
import json
import os

from pymongo import MongoClient

from analyser.text_normalize import normalize_company_name


def convert_initialData():
  jfile = "/Users/artem/work/nemo/goil/gpn-ui/projects/server/initialData.json"

  forms = []

  converted = {}
  result = {'Subsidiary': converted}
  with open(jfile, 'r', encoding='utf8') as handle:
    subsidiaries = json.load(handle)['Subsidiary']

  for s in subsidiaries:
    legal_entity_type, name = normalize_company_name(s['name'])
    if legal_entity_type not in forms:
      forms.append(legal_entity_type)
    # print(f'{legal_entity_type}\t"{name}"')
    converted[name] = {
      'legal_entity_type': legal_entity_type,
      'aliases': [name],
      '_id': name
    }

  return result


if __name__ == '__main__':
  data = list(convert_initialData()['Subsidiary'].values())
  # data = sorted(data, key='id_')
  # print(data['Subsidiary'])
  if 'GPN_DB_NAME' not in os.environ:
    os.environ['GPN_DB_NAME'] = 'gpn'
  print(os.environ['GPN_DB_NAME'])

  client = MongoClient('mongodb://localhost:27017/')
  db = client[os.environ['GPN_DB_NAME']]
  db.drop_collection('Subsidiary')
  db['Subsidiary'].insert_many(data)

  # test read:
  all = db['Subsidiary'].find({})

  for document in all:
    print(document['legal_entity_type'], '\t', document['aliases'])
  # //pprint(serverStatusResult)
