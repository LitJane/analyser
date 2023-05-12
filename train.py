#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


from trainsets.retrain_contract_uber_model import UberModelTrainsetManager, default_work_dir

if __name__ == '__main__':
  umtm = UberModelTrainsetManager(default_work_dir)
  umtm.run()

