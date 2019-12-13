import pymongo
import numpy as np
import textdistance

import analyser
from integration.db import get_mongodb_connection

currency_rates = {"RUB": 1.0, "USD": 63.72, "EURO": 70.59, "KZT": 0.17}


def extract_text(span, words, text):
    first_idx = words[span[0]][0]
    last_idx = words[span[1]][0] - 1
    return text[first_idx:last_idx]


def get_attrs(document):
    attrs = document["analysis"]["attributes"]
    if document.get("user") is not None:
        attrs = document["user"]["attributes"]
    return attrs


def get_docs_by_audit_id(id: str, state, kind=None):
    db = get_mongodb_connection()
    documents_collection = db['documents']

    query = {
        'auditId': id,
        'parse.documentType': kind,
        "state": state,
        "analysis.attributes.date": {"$ne": None}
    }

    res = documents_collection.find(query).sort([("analysis.attributes.date.value", pymongo.ASCENDING)])
    return res


def save_violations(audit, violations):
    audit["violations"] = violations
    db = get_mongodb_connection()
    db["audits"].update_one({'_id': audit["_id"]}, {"$set": {"violations": violations}})


def create_violation(document_id, founding_document_id, reference, violation_type, violation_reason):
    return {"document": document_id, "founding_document": founding_document_id, "reference": reference, "violation_type": violation_type, "violation_reason": violation_reason}


def convert_to_rub(value_currency):
    value_currency["original_value"] = value_currency["value"]
    value_currency["original_currency"] = value_currency["currency"]
    value_currency["value"] = currency_rates.get(value_currency["currency"], "RUB") * value_currency["value"]
    value_currency["currency"] = "RUB"
    return value_currency


def get_constraints_rub(key, attributes):
    constraints = []
    for key2, value2 in attributes.items():
        if value2.get("parent") is not None and value2["parent"] == key:
            result = {}
            for key3, value3 in attributes.items():
                if value3.get("parent") is not None and value3["parent"] == key2:
                    if key3.endswith("sign"):
                        result["sign"] = value3["value"]
                    elif key3.endswith("value"):
                        result["value"] = value3["value"]
                    elif key3.endswith("currency"):
                        result["currency"] = value3["value"]
            constraints.append(result)
    for constraint in constraints:
        convert_to_rub(constraint)
    return constraints


def get_charter_diapasons(charter):
    #group by subjects
    subjects = {}
    charter_attrs = get_attrs(charter)
    min_constraint = np.inf
    for key, value in charter_attrs.items():
        if key.count("/") == 1:
            subject_type = value["value"]
            subject_map = subjects.get(subject_type)
            if subject_map is None:
                subject_map = {}
                subjects[subject_type] = subject_map
            constraints = get_constraints_rub(key, charter_attrs)
            for constraint in constraints:
                if subject_map.get(value["parent"]) is None:
                    subject_map[value["parent"]] = {"min": 0, "max": np.inf, "competence_attr_name": key}
                if constraint["sign"] > 0:
                    if subject_map[value["parent"]]["min"] == 0:
                        subject_map[value["parent"]]["min"] = constraint["value"]
                        subject_map[value["parent"]]["original_min"] = constraint["original_value"]
                        subject_map[value["parent"]]["original_currency_min"] = constraint["original_currency"]
                    else:
                        old_value = subject_map[value["parent"]]["min"]
                        new_value = constraint["value"]
                        if new_value < old_value:
                            subject_map[value["parent"]]["min"] = constraint["value"]
                            subject_map[value["parent"]]["original_min"] = constraint["original_value"]
                            subject_map[value["parent"]]["original_currency_min"] = constraint["original_currency"]
                    min_constraint = min(min_constraint, constraint["value"])
                else:
                    if subject_map[value["parent"]]["max"] == np.inf:
                        subject_map[value["parent"]]["max"] = constraint["value"]
                        subject_map[value["parent"]]["original_max"] = constraint["original_value"]
                        subject_map[value["parent"]]["original_currency_max"] = constraint["original_currency"]
                    else:
                        old_value = subject_map[value["parent"]]["max"]
                        new_value = constraint["value"]
                        if new_value > old_value:
                            subject_map[value["parent"]]["max"] = constraint["value"]
                            subject_map[value["parent"]]["original_max"] = constraint["original_value"]
                            subject_map[value["parent"]]["original_currency_max"] = constraint["original_currency"]
    return subjects, min_constraint


def clean_name(name):
    return name.replace(" ", "").replace("-", "").replace("_", "").lower()


def find_protocol(contract, protocols, org_level, audit):
    contract_attrs = get_attrs(contract)
    result = []
    for protocol in protocols:
        protocol_attrs = get_attrs(protocol)
        if protocol_attrs["org_structural_level"]["value"] == org_level:
            for protocol_key, protocol_value in protocol_attrs.items():
                if protocol_key.endswith("-name"):
                    for contract_key, contract_value in contract_attrs.items():
                        if contract_key.endswith("-name") and contract_value["value"] != audit["subsidiary"]["name"]:
                            clean_protocol_org = clean_name(protocol_value["value"])
                            clean_contract_org = clean_name(contract_value["value"])
                            distance = textdistance.levenshtein.normalized_distance(clean_contract_org, clean_protocol_org)
                            if distance > 0.9:
                                result.append(protocol)
    if len(result) == 0:
        return None
    else:
        return result[0]


def check_contract(contract, charters, protocols, audit):
    violations = []
    contract_attrs = get_attrs(contract)
    eligible_charter = None
    for charter in charters:
        charter_attrs = get_attrs(charter)
        if charter_attrs["date"]["value"] <= contract_attrs["date"]["value"]:
            eligible_charter = charter
            break

    if eligible_charter is None:
        json_charters = []
        for charter in charters:
            charter_attrs = get_attrs(charter)
            json_charters.append({"id": charter["_id"], "date": charter_attrs["date"]["value"]})
        violations.append(create_violation({"id": contract["_id"], "number": contract_attrs["number"]["value"], "type": contract["parse"]["documentType"]},
                                           None,
                                           None,
                                           "charter_not_found",
                                           {"contract": {"id": contract["_id"], "number": contract_attrs["number"]["value"], "type": contract["parse"]["documentType"], "date": contract_attrs["date"]["value"]},
                                            "charters": json_charters}))
        return violations
    else:
        charter_subject_map, min_constraint = get_charter_diapasons(eligible_charter)
        eligible_charter_attrs = get_attrs(eligible_charter)
        competences = charter_subject_map.get(contract_attrs["subject"]["value"])
        contract_value = convert_to_rub({"value": contract_attrs["sign_value_currency/value"]["value"], "currency": contract_attrs["sign_value_currency/currency"]["value"]})
        if competences is not None:
            eligible_protocol = None
            need_protocol_check = False
            competence_constraint = None
            for competence, constraint in competences.items():
                if constraint["min"] <= contract_value["value"] <= constraint["max"]:
                    need_protocol_check = True
                    competence_constraint = constraint
                    eligible_protocol = find_protocol(contract, protocols, competence, audit)

            attribute = None
            text = None
            min_value = None
            max_value = None
            if competence_constraint is not None:
                attribute = competence_constraint.get("competence_attr_name")
                if attribute is not None:
                    text = extract_text(eligible_charter_attrs[attribute]["span"],
                                        eligible_charter["analysis"]["tokenization_maps"]["words"],
                                        eligible_charter["analysis"]["normal_text"])
                if competence_constraint["min"] != 0:
                    min_value = {"value": competence_constraint["original_min"], "currency": competence_constraint["original_currency_min"]}
                if competence_constraint["max"] != np.inf:
                    max_value = {"value": competence_constraint["original_max"], "currency": competence_constraint["original_currency_max"]}

            if eligible_protocol is not None:
                eligible_protocol_attrs = get_attrs(eligible_protocol)
                if eligible_protocol_attrs["date"]["value"] > contract_attrs["date"]["value"]:
                    violations.append(create_violation(
                        {"id": contract["_id"], "number": contract_attrs["number"]["value"],
                         "type": contract["parse"]["documentType"]},
                        {"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]},
                        {"id": eligible_charter["_id"], "attribute": attribute, "text": text},
                        "contract_date_less_than_protocol_date",
                        {"contract": {"number": contract_attrs["number"]["value"],
                                      "date": contract_attrs["date"]["value"],
                                      "org_type": contract_attrs["org-2-type"]["value"],
                                      "org_name": contract_attrs["org-2-name"]["value"]},
                         "protocol": {"org_structural_level": eligible_protocol_attrs["org_structural_level"]["value"],
                                      "date": eligible_protocol_attrs["date"]["value"]}}))
                else:
                    for key, value in eligible_protocol_attrs.items():
                        if key.endswith("/value"):
                            converted_value = convert_to_rub({"value": value["value"], "currency": eligible_protocol_attrs[key[:-5] + "currency"]["value"]})
                            if min_constraint <= converted_value["value"] < contract_value["value"]:
                                violations.append(create_violation(
                                    {"id": contract["_id"], "number": contract_attrs["number"]["value"],
                                     "type": contract["parse"]["documentType"]},
                                    {"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]},
                                    {"id": eligible_charter["_id"], "attribute": attribute, "text": text},
                                    "contract_value_great_than_protocol_value",
                                    {"contract": {"number": contract_attrs["number"]["value"],
                                                  "date": contract_attrs["date"]["value"],
                                                  "org_type": contract_attrs["org-2-type"]["value"],
                                                  "org_name": contract_attrs["org-2-name"]["value"],
                                                  "value": contract_attrs["sign_value_currency/value"]["value"],
                                                  "currency": contract_attrs["sign_value_currency/currency"]["value"]},
                                     "protocol": {
                                         "org_structural_level": eligible_protocol_attrs["org_structural_level"][
                                             "value"], "date": eligible_protocol_attrs["date"]["value"]}}))
                                break
            else:
                if need_protocol_check:
                    violations.append(create_violation(
                        {"id": contract["_id"], "number": contract_attrs["number"]["value"],
                         "type": contract["parse"]["documentType"]},
                        {"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]},
                        {"id": eligible_charter["_id"], "attribute": attribute, "text": text},
                        {"type": "protocol_not_found", "subject": contract_attrs["subject"]["value"],
                         "org_structural_level": eligible_charter_attrs[eligible_charter_attrs[attribute]["parent"]]["value"],
                         "min": min_value,
                         "max": max_value
                         },
                        {"contract": {"number": contract_attrs["number"]["value"],
                                      "date": contract_attrs["date"]["value"],
                                      "org_type": contract_attrs["org-2-type"]["value"],
                                      "org_name": contract_attrs["org-2-name"]["value"],
                                      "value": contract_attrs["sign_value_currency/value"]["value"],
                                      "currency": contract_attrs["sign_value_currency/currency"]["value"]}}))
    return violations


def finalize(audit):
    violations = []
    contracts = get_docs_by_audit_id(audit["_id"], 15, "CONTRACT")
    charters = sorted(get_docs_by_audit_id(audit["_id"], 15, "CHARTER"), key=lambda k: get_attrs(k)["date"]["value"])
    protocols = get_docs_by_audit_id(audit["_id"], 15, "PROTOCOL")

    for contract in contracts:
        violations.extend(check_contract(contract, charters, protocols, audit))

    save_violations(audit, violations)


def create_fake_finalization(audit):
    violations = []
    contracts = get_docs_by_audit_id(audit["_id"], 15, "CONTRACT")
    charters = sorted(get_docs_by_audit_id(audit["_id"], 15, "CHARTER"), key=lambda k: get_attrs(k)["date"]["value"])
    protocols = get_docs_by_audit_id(audit["_id"], 15, "PROTOCOL")

    for contract in contracts:
        eligible_protocol = next(protocols)
        eligible_charter = charters[0]
        contract_attrs = get_attrs(contract)
        eligible_charter_attrs = get_attrs(eligible_charter)
        eligible_protocol_attrs = get_attrs(eligible_protocol)
        violations.append(create_violation({"id": contract["_id"], "number": contract_attrs["number"]["value"], "type": contract["parse"]["documentType"]},
                                           None,
                                           None,
                                           "charter_not_found",
                                           {"contract": {"id": contract["_id"], "number": contract_attrs["number"]["value"], "type": contract["parse"]["documentType"], "date": contract_attrs["date"]["value"]},
                                            "charters": [{"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]}]}))
        violations.append(create_violation({"id": contract["_id"], "number": contract_attrs["number"]["value"], "type": contract["parse"]["documentType"]},
                                           {"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]},
                                           {"id": eligible_charter["_id"], "attribute": "BoardOfDirectors/Deal-3",
                                            "text": extract_text(eligible_charter_attrs["BoardOfDirectors/Deal-3"]["span"], eligible_charter["analysis"]["tokenization_maps"]["words"], eligible_charter["analysis"]["normal_text"])},
                                           "contract_date_less_than_protocol_date",
                                           {"contract": {"number": contract_attrs["number"]["value"], "date": contract_attrs["date"]["value"],
                                            "org_type": contract_attrs["org-2-type"]["value"], "org_name": contract_attrs["org-2-name"]["value"]},
                                            "protocol": {"org_structural_level": eligible_protocol_attrs["org_structural_level"]["value"], "date": eligible_protocol_attrs["date"]["value"]}}))
        violations.append(create_violation({"id": contract["_id"], "number": contract_attrs["number"]["value"], "type": contract["parse"]["documentType"]},
                                           {"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]},
                                           {"id": eligible_charter["_id"], "attribute": "BoardOfDirectors/Deal-3",
                                            "text": extract_text(eligible_charter_attrs["BoardOfDirectors/Deal-3"]["span"], eligible_charter["analysis"]["tokenization_maps"]["words"], eligible_charter["analysis"]["normal_text"])},
                                           "contract_value_great_than_protocol_value",
                                           {"contract": {"number": contract_attrs["number"]["value"],
                                                         "date": contract_attrs["date"]["value"],
                                                         "org_type": contract_attrs["org-2-type"]["value"],
                                                         "org_name": contract_attrs["org-2-name"]["value"],
                                                         "value": contract_attrs["sign_value_currency/value"]["value"],
                                                         "currency": contract_attrs["sign_value_currency/currency"]["value"]},
                                            "protocol": {
                                                "org_structural_level": eligible_protocol_attrs["org_structural_level"][
                                                    "value"], "date": eligible_protocol_attrs["date"]["value"]}}))
        violations.append(create_violation({"id": contract["_id"], "number": contract_attrs["number"]["value"], "type": contract["parse"]["documentType"]},
                                           {"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]},
                                           {"id": eligible_charter["_id"], "attribute": "BoardOfDirectors/Deal-3",
                                            "text": extract_text(eligible_charter_attrs["BoardOfDirectors/Deal-3"]["span"], eligible_charter["analysis"]["tokenization_maps"]["words"], eligible_charter["analysis"]["normal_text"])},
                                           {"type": "protocol_not_found", "subject": contract_attrs["subject"]["value"], "org_structural_level": eligible_protocol_attrs["org_structural_level"]["value"],
                                            "min": {"value": eligible_protocol_attrs["agenda_item_1/sign_value_currency-2/value"]["value"],
                                            "currency": eligible_protocol_attrs["agenda_item_1/sign_value_currency-2/currency"]["value"]},
                                            "max": {"value": eligible_protocol_attrs["agenda_item_1/sign_value_currency-2/value"]["value"],
                                                    "currency": eligible_protocol_attrs["agenda_item_1/sign_value_currency-2/currency"]["value"]}
                                            },
                                           {"contract": {"number": contract_attrs["number"]["value"],
                                                         "date": contract_attrs["date"]["value"],
                                                         "org_type": contract_attrs["org-2-type"]["value"],
                                                         "org_name": contract_attrs["org-2-name"]["value"],
                                                         "value": contract_attrs["sign_value_currency/value"]["value"],
                                                         "currency": contract_attrs["sign_value_currency/currency"]["value"]}}))
    save_violations(audit, violations)


if __name__ == '__main__':
    db = get_mongodb_connection()
    audits_collection = db['audits']
    audits = audits_collection.find({'status': 'Finalizing'}).sort([("createDate", pymongo.ASCENDING)])
    for audit in audits:
        finalize(audit)
        # create_fake_finalization(audit)
