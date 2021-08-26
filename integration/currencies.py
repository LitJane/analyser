import json
import os
from datetime import datetime

import requests
import urllib3

from analyser.log import logger

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
currency_rates = {"RUB": 1.0, "USD": 63.72, "EUR": 70.59, "KZT": 0.17}


def _env_var(vname, default_val=None):
    if vname not in os.environ:
        msg = f'Currency-service : define {vname} environment variable! defaulting to {default_val}'
        logger.warning(msg)
        return default_val
    else:
        return os.environ[vname]


user = _env_var('GPN_CURRENCY_USER')
password = _env_var('GPN_CURRENCY_PASSWORD')
url = _env_var('GPN_CURRENCY_URL')


def convert_to_currency(value_currency, new_currency, date=None):
    value_currency["original_value"] = value_currency["value"]
    value_currency["original_currency"] = value_currency["currency"]
    if new_currency != value_currency['currency']:
        value_currency["value"] = _get_rate(value_currency["original_currency"], new_currency, date) * float(value_currency["value"])
    value_currency["currency"] = new_currency
    return value_currency


def _get_rate(original_currency, new_currency, date):
    rate = 1.0
    if user is not None and password is not None and url is not None:
        try:
            rate = _get_rate_online(original_currency, new_currency, date)
        except Exception as e:
            logger.exception(e)
    else:
        rate = _get_rate_offline(original_currency, new_currency)
    return rate


def _get_rate_offline(original_currency, new_currency):
    return currency_rates.get(original_currency, 1.0) / currency_rates.get(new_currency, 1.0)


def _get_rate_online(original_currency, new_currency, date):
    if date is None:
        date_str = datetime.today().strftime('%Y-%m-%d')
    else:
        date_str = date.strftime('%Y-%m-%d')
    response = requests.post(url, data=f'{{"Date": "{date_str}T00:00:00.0Z"}}', auth=(user, password), verify=False)
    if response.status_code != 200:
        raise RuntimeError(f'Currency service response code: {response.status_code}. Error message: {response.reason}')
    else:
        json_result = json.loads(response.content)
        original_rate = 1.0
        if original_currency != 'RUB':
            original_rate_str = list(filter(lambda el: el['IsoCharCode'] == original_currency, json_result['item']))[0]['Rate']
            original_rate = float(original_rate_str.strip())
        new_rate = 1.0
        if new_currency != 'RUB':
            new_rate_str = list(filter(lambda el: el['IsoCharCode'] == new_currency, json_result['item']))[0]['Rate']
            new_rate = float(new_rate_str.strip())
        return original_rate / new_rate
