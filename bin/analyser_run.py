#!/usr/bin/env python
import os
import time

import schedule

from analyser import runner
from analyser.attributes import convert_all_docs
from analyser.dictionaries import update_db_dictionaries
from integration.csgk import sync_csgk_data


def migrate():
  # 1:
  convert_all_docs()
  # 2:
  # something else


def main():
  update_db_dictionaries()
  migrate()

  check_interval = os.environ.get("GPN_DB_CHECK_INTERVAL")
  if check_interval is None:
    check_interval = 30
    print("Environment variable GPN_DB_CHECK_INTERVAL not set. Default value is %d sec." % (check_interval))

  schedule.every(int(check_interval)).seconds.do(runner.run)
  schedule.every().day.at("03:03").do(sync_csgk_data)
  sync_csgk_data()
  runner.run()
  while True:
    schedule.run_pending()
    time.sleep(1)


if __name__ == '__main__':
  main()
