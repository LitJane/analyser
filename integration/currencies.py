

currency_rates = {"RUB": 1.0, "USD": 63.72, "EURO": 70.59, "KZT": 0.17}


def convert_to_currency(value_currency, new_currency, date=None):
    value_currency["original_value"] = value_currency["value"]
    value_currency["original_currency"] = value_currency["currency"]
    value_currency["currency"] = new_currency
    if new_currency != value_currency['currency']:
        value_currency["value"] = get_rate(value_currency["original_currency"], new_currency) * float(value_currency["value"])
    return value_currency


def get_rate(original_currency, new_currency):
    return currency_rates.get(original_currency, 1.0) / currency_rates.get(new_currency, 1.0)
