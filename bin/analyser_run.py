#!/usr/bin/env python
import time

import schedule

from analyser import runner
from analyser.attributes import convert_all_docs
from analyser.dictionaries import update_db_dictionaries
from gpn_config import configured
from integration.csgk import sync_csgk_data


def migrate():
  # 1:
  convert_all_docs()
  # 2:
  # something else


def main():
  update_db_dictionaries()
  migrate()

  check_interval = configured("GPN_DB_CHECK_INTERVAL", 30)

  schedule.every(int(check_interval)).seconds.do(runner.run)
  schedule.every().day.at("03:03").do(sync_csgk_data)
  sync_csgk_data()
  runner.run()
  while True:
    schedule.run_pending()
    time.sleep(1)


if __name__ == '__main__':
  main()
