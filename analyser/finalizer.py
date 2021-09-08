import json
import re
from collections import deque
from types import SimpleNamespace

import numpy as np
import pymongo
import textdistance
from bson import ObjectId

from analyser.log import logger
from analyser.structures import legal_entity_types
from analyser.text_normalize import normalize_company_name
from integration import mail
from integration.currencies import convert_to_currency
from integration.db import get_mongodb_connection


full_name_pattern = re.compile(r'(?P<last_name>[а-я,А-Я,a-z,A-Z]+) +(?P<first_name>[а-я,А-Я,a-z,A-Z]+)(\.? +)?(?P<middle_name>[а-я,А-Я,a-z,A-Z]+)?')
company_name_pattern = re.compile(r'[«\'"](?P<company_name>.+)[»\'"]')
companies = {'gp': 'Газпром', 'gpn': 'Газпром нефть'}


def normalize_only_company_name(name: str) -> str:
    _, result = normalize_company_name(name)
    return result


def compare_ignore_case(str1: str, str2: str) -> bool:
    if str1 is None or str2 is None:
        return False
    return str1.lower() == str2.lower()


def get_audits():
    db = get_mongodb_connection()
    audits_collection = db['audits']

    res = audits_collection.find({'status': 'Finalizing'}).sort([("createDate", pymongo.ASCENDING)])
    return res


def remove_old_links(audit_id, contract_id):
    db = get_mongodb_connection()
    audit_collection = db['audits']
    audit_collection.update_one({"_id": audit_id}, {"$pull": {"links": {"type": "analysis", "$or": [{"toId": contract_id}, {"fromId": contract_id}]}}})


def get_linked_docs(audit, contract_id):
    db = get_mongodb_connection()
    result = []
    document_collection = db['documents']
    for link in audit['links']:
        if link['fromId'] == contract_id:
            result.append(document_collection.find_one({'_id': link['toId']}))
        elif link['toId'] == contract_id:
            result.append(document_collection.find_one({'_id': link['fromId']}))

    return result


def add_link(audit_id, doc_id1, doc_id2):
    db = get_mongodb_connection()
    audit_collection = db['audits']
    audit_collection.update_one({"_id": audit_id}, {"$push": {"links": {"fromId": doc_id1, "toId": doc_id2, "type": "analysis"}}})


def change_contract_primary_subject(contract, new_subject):
    db = get_mongodb_connection()
    db['documents'].update_one({'_id': contract['_id']}, {'$set': {'primary_subject': new_subject}})


def get_book_value(audit, target_year: str):
    for record in audit['bookValues']:
        if record.get(target_year) is not None:
            return float(record[target_year])
    return None


def extract_text(span, words, text):
    first_idx = words[span[0]][0]
    last_idx = words[span[1]][0]
    return text[first_idx:last_idx]


def get_nearest_header(headers, position):
    found_header = headers[0]
    for header in headers:
        if header["span"][0] > position:
            return found_header
        else:
            found_header = header
    return found_header


def get_attrs(document):
    attrs = document["analysis"]["attributes_tree"]
    if document.get("user") is not None:
        attrs = document["user"]["attributes_tree"]
    return next(iter(attrs.values()))


def get_docs_by_audit_id(id: str, state, kind=None, id_only=False, without_large_fields=False):
    db = get_mongodb_connection()
    documents_collection = db['documents']

    query = {
        'auditId': id,
        "state": state,
        # "$or": [
        #     {"$and": [{"analysis.attributes.date": {"$ne": None}}, {"user": None}]},
        #     {"user.attributes.date": {"$ne": None}}
        # ]
    }
    if kind is not None:
        query['documentType'] = kind
    if id_only:
        res = documents_collection.find(query, projection={'_id': True})
    else:
        if without_large_fields:
            res = documents_collection.find(query, projection={'analysis.original_text': False,
                                                               'analysis.normal_text': False,
                                                               'analysis.tokenization_maps': False,
                                                               'analysis.headers': False,
                                                               'parse.paragraphs': False})
        else:
            res = documents_collection.find(query)
    docs = []
    for doc in res:
        if not id_only and get_attrs(doc).get('date') is not None:
            docs.append(doc)
        else:
            docs.append(doc)
    return docs


def get_doc_by_id(doc_id:ObjectId):
    db = get_mongodb_connection()
    documents_collection = db['documents']
    return documents_collection.find_one({'_id': doc_id})


def get_audit_by_id(aid:ObjectId):
    db = get_mongodb_connection()
    return db['audits'].find_one({'_id': aid})


def save_violations(audit, violations):
    db = get_mongodb_connection()
    db['audits'].update_one({'_id': audit['_id']}, {'$pull': {'violations': {'userViolation': False}}})
    db["audits"].update_one({'_id': audit["_id"]}, {"$push": {"violations": {'$each': violations}}})
    db["audits"].update_one({'_id': audit["_id"]}, {"$set": {"status": "Done"}})


def create_violation(document_id, founding_document_id, reference, violation_type, violation_reason):
    return {'id': ObjectId(), 'userViolation': False, "document": document_id, "founding_document": founding_document_id, "reference": reference, "violation_type": violation_type, "violation_reason": violation_reason}


def get_charter_diapasons(charter):
    #group by subjects
    subjects = {}
    charter_attrs = get_attrs(charter)
    min_constraint = np.inf
    charter_currency = 'RUB'
    if charter_attrs.get('structural_levels') is not None:
        for structural_level in charter_attrs['structural_levels']:
            if structural_level.get('value') is not None:
                structural_level_name = structural_level['value']
                if structural_level.get('competences') is not None:
                    for competence in structural_level['competences']:
                        if competence.get('value') is None:
                            continue
                        subject_type = competence['value']
                        subject_map = subjects.get(subject_type)
                        if subject_map is None:
                            subject_map = {}
                            subjects[subject_type] = subject_map
                        if subject_map.get(structural_level_name) is None:
                            subject_map[structural_level_name] = {"min": 0, "max": np.inf, "competence_attr_name": competence.get('span')}
                        constraints = competence.get('constraints')
                        if constraints is not None:
                            if len(constraints) == 0:
                                min_constraint = 0
                            for constraint in constraints:
                                constraint_currency = constraint['currency']['value']
                                constraint_amount = constraint['amount']['value']
                                if constraint_currency != 'Percent':
                                    charter_currency = constraint_currency
                                if constraint.get('sign') is not None and int(constraint['sign'].get("value", 0)) > 0:
                                    if subject_map[structural_level_name]["min"] == 0:
                                        subject_map[structural_level_name]["min"] = constraint_amount
                                        subject_map[structural_level_name]["currency_min"] = constraint_currency
                                    else:
                                        old_value = subject_map[structural_level_name]["min"]
                                        if constraint_amount < old_value:
                                            subject_map[structural_level_name]["min"] = constraint_amount
                                            subject_map[structural_level_name]["currency_min"] = constraint_currency
                                    min_constraint = min(min_constraint, constraint_amount)
                                else:
                                    if subject_map[structural_level_name]["max"] == np.inf:
                                        subject_map[structural_level_name]["max"] = constraint_amount
                                        subject_map[structural_level_name]["currency_max"] = constraint_currency
                                    else:
                                        old_value = subject_map[structural_level_name]["max"]
                                        if constraint_amount > old_value:
                                            subject_map[structural_level_name]["max"] = constraint_amount
                                            subject_map[structural_level_name]["currency_max"] = constraint_currency
    if min_constraint == np.inf:
        min_constraint = 0
    return subjects, min_constraint, charter_currency


def clean_name(name):
    return name.replace(" ", "").replace("-", "").replace("_", "").lower()


def find_protocol(contract, protocols, org_level, contract_value, check_orgs=True):
    contract_attrs = get_attrs(contract)
    result = None
    best_value = {'value': -1, 'currency': 'RUB'}
    best_sign = 0
    clean_contract_orgs=[]
    for contract_org in contract_attrs['orgs']:
        if contract_org.get('name') is not None and contract_org['name'].get('value') is not None:
            clean_contract_orgs.append(clean_name(contract_org['name']['value']))
    for protocol in protocols:
        protocol_attrs = get_attrs(protocol)
        if protocol_attrs.get("structural_level") is not None and protocol_attrs["structural_level"]["value"] == org_level and protocol_attrs.get('agenda_items') is not None:
            for agenda_item in protocol_attrs['agenda_items']:
                if agenda_item.get('contracts') is not None:
                    for agenda_contract in agenda_item['contracts']:
                        if agenda_contract.get('orgs') is not None and contract_attrs.get('orgs') is not None:
                            for agenda_org in agenda_contract['orgs']:
                                if agenda_org.get('name') is not None and agenda_org['name'].get('value') is not None:
                                    clean_protocol_org = clean_name(agenda_org['name']["value"])
                                    for clean_contract_org in clean_contract_orgs:
                                        distance = textdistance.levenshtein.normalized_distance(clean_contract_org, clean_protocol_org)
                                        if distance < 0.1 or not check_orgs:
                                            if contract_value is not None and agenda_contract.get('price') is not None:
                                                protocol_value = convert_to_currency({'value': agenda_contract['price']['amount']['value'], 'currency': agenda_contract['price']['currency']['value']}, contract_value['currency'])
                                                if contract_value['value'] <= best_value['value']:
                                                    if best_value['value'] >= protocol_value['value'] >= contract_value['value']:
                                                        result = protocol
                                                        best_value = protocol_value
                                                        sign = 0
                                                        if agenda_contract['price'].get('sign') is not None:
                                                            sign = agenda_contract['price']['sign']['value']
                                                        best_sign = sign
                                                else:
                                                    if protocol_value['value'] > best_value['value']:
                                                        result = protocol
                                                        best_value = protocol_value
                                                        sign = 0
                                                        if agenda_contract['price'].get('sign') is not None:
                                                            sign = agenda_contract['price']['sign']['value']
                                                        best_sign = sign
                                            else:
                                                if best_value['value'] == -1:
                                                    result = protocol

    return result, best_value, best_sign


def find_supplementary_agreements(contract, sup_agreements, audit):
    contract_attrs = get_attrs(contract)
    result = []
    if contract_attrs.get('number') is not None:
        contract_number = contract_attrs['number']['value']
    else:
        return result
    for sup_agreement in sup_agreements:
        sup_agreement_attrs = get_attrs(sup_agreement)
        if sup_agreement_attrs.get('number') is not None and sup_agreement_attrs['number']['value'] == contract_number:
            result.append(sup_agreement)
            add_link(audit['_id'], contract['_id'], sup_agreement['_id'])
    return result


def get_org(doc_attrs):
    if doc_attrs.get('org') is not None:
        return doc_attrs['org']
    elif doc_attrs.get('orgs') is not None and len(doc_attrs['orgs']) > 0:
        return doc_attrs['orgs'][0]
    else:
        return None


def get_charter_span(charter_atts, org_level, subject):
    if charter_atts.get('structural_levels') is not None:
        for structural_level in charter_atts['structural_levels']:
            if org_level == structural_level['value'] and structural_level.get('competences') is not None:
                for competence in structural_level['competences']:
                    if competence.get('value') is not None and subject == competence['value']:
                        return competence['span']
    return None


def check_contract(contract, charters, protocols, audit, supplementary_agreements):
    violations = []
    contract_attrs = get_attrs(contract)
    contract_number = ""
    remove_old_links(audit["_id"], contract["_id"])
    user_linked_docs = get_linked_docs(audit, contract["_id"])
    if contract_attrs.get("number") is not None:
        contract_number = contract_attrs["number"]["value"]
        # linked_sup_agreements = list(filter(lambda doc: doc['documentType'] == 'SUPPLEMENTARY_AGREEMENT', user_linked_docs))
        find_supplementary_agreements(contract, supplementary_agreements, audit)
    eligible_charter = None
    linked_charters = list(filter(lambda doc: doc['documentType'] == 'CHARTER', user_linked_docs))
    if linked_charters:
        eligible_charter = linked_charters[0]
    else:
        for charter in charters:
            charter_attrs = get_attrs(charter)
            if charter_attrs["date"]["value"] <= contract_attrs["date"]["value"]:
                eligible_charter = charter
                add_link(audit["_id"], contract["_id"], eligible_charter["_id"])
                break

    if eligible_charter is None:
        json_charters = []
        for charter in charters:
            charter_attrs = get_attrs(charter)
            json_charters.append({"id": charter["_id"], "date": charter_attrs["date"]["value"]})

        violation_reason = {"contract":
                              {"id": contract["_id"],
                               "number": contract_number,
                               "type": contract["documentType"],
                               'date': contract_attrs["date"]["value"]
                               },
                            "charters": json_charters
                            }

        violations.append(create_violation(
          document_id={
            "id": contract["_id"],
            "number": contract_number,
            "type": contract["documentType"]
          },
          founding_document_id=None,
          reference=None,
          violation_type="charter_not_found",
          violation_reason=violation_reason)
        )
        return violations
    else:
        charter_subject_map, min_constraint, charter_currency = get_charter_diapasons(eligible_charter)
        eligible_charter_attrs = get_attrs(eligible_charter)
        competences = None
        if contract_attrs.get("subject") is not None:
            competences = charter_subject_map.get(contract_attrs["subject"]["value"])
        if competences is None:
            competences = charter_subject_map.get("Deal")
        contract_value = None
        book_value = None
        if audit.get('bookValues') is not None:
            book_value = get_book_value(audit, str(contract_attrs["date"]["value"].year - 1))
        contract_value = get_amount_netto(contract_attrs.get('price'))
        if contract_value is not None:
            contract_value = convert_to_currency(contract_value, charter_currency)

            if contract_value is not None and book_value is not None:
                org = get_org(eligible_charter_attrs)
                if org is not None and org.get('type') is not None and 'акционерное общество' == org['type']['value'].lower():
                    if book_value * 0.25 < contract_value["value"] <= book_value * 0.5:
                        competences = {'BoardOfDirectors': {"min": 25, "currency_min": "Percent", "max": 50, "currency_max": "Percent", "competence_attr_name": get_charter_span(eligible_charter_attrs, 'BoardOfDirectors', 'BigDeal')}}
                        change_contract_primary_subject(contract, 'BigDeal')
                    elif contract_value["value"] > book_value * 0.5:
                        competences = {'ShareholdersGeneralMeeting': {"min": 50, "currency_min": "Percent", "max": np.inf, "competence_attr_name": get_charter_span(eligible_charter_attrs, 'ShareholdersGeneralMeeting', 'BigDeal')}}
                        change_contract_primary_subject(contract, 'BigDeal')
                else:
                    if charter_subject_map.get('BigDeal') is not None:
                        if charter_subject_map['BigDeal'].get('BoardOfDirectors') is not None:
                            big_deal_subject_charter_competence = charter_subject_map['BigDeal'].get('BoardOfDirectors')
                            if big_deal_subject_charter_competence.get('min') is not None \
                                    and big_deal_subject_charter_competence.get('currency_min') is not None:
                                limit = big_deal_subject_charter_competence['min']
                                if big_deal_subject_charter_competence['currency_min'] == 'Percent':
                                    limit = big_deal_subject_charter_competence['min'] * book_value / 100
                                if contract_value['value'] > limit:
                                    change_contract_primary_subject(contract, 'BigDeal')
                                    competences = {'BoardOfDirectors': big_deal_subject_charter_competence}
                        if charter_subject_map['BigDeal'].get('AllMembers') is not None:
                            big_deal_subject_charter_competence = charter_subject_map['BigDeal'].get('AllMembers')
                            if big_deal_subject_charter_competence.get('min') is not None \
                                    and big_deal_subject_charter_competence.get('currency_min') is not None:
                                limit = big_deal_subject_charter_competence['min']
                                if big_deal_subject_charter_competence['currency_min'] == 'Percent':
                                    limit = big_deal_subject_charter_competence['min'] * book_value / 100
                                if contract_value['value'] > limit:
                                    change_contract_primary_subject(contract, 'BigDeal')
                                    competences = {'AllMembers': big_deal_subject_charter_competence}

        if competences is not None and contract_value is not None:
            eligible_protocol = None
            need_protocol_check = False
            competence_constraint = None
            org_level = None
            protocol_value = None
            sign = None

            for competence, constraint in competences.items():
                constraint_currency_min = constraint.get('currency_min')
                constraint_currency_max = constraint.get('currency_max')
                if constraint_currency_min is not None and constraint_currency_min == 'Percent' and book_value is not None:
                    abs_min = constraint['min'] * book_value / 100
                else:
                    abs_min = constraint['min']
                if constraint_currency_max is not None and constraint_currency_max == 'Percent' and book_value is not None:
                    abs_max = constraint['max'] * book_value / 100
                else:
                    abs_max = constraint['max']

                if abs_min <= contract_value["value"] <= abs_max:
                    need_protocol_check = True
                    competence_constraint = constraint
                    linked_protocols = list(filter(lambda doc: doc['documentType'] == 'PROTOCOL', user_linked_docs))
                    if linked_protocols:
                        eligible_protocol, protocol_value, sign = find_protocol(contract, linked_protocols, competence, contract_value)
                        if eligible_protocol is None: #force find protocol_value and sign
                            eligible_protocol, protocol_value, sign = find_protocol(contract, linked_protocols, competence, contract_value, check_orgs=False)
                    else:
                        eligible_protocol, protocol_value, sign = find_protocol(contract, protocols, competence, contract_value)
                        if eligible_protocol is not None:
                            add_link(audit["_id"], contract["_id"], eligible_protocol["_id"])
                    if eligible_protocol is not None:
                        org_level = competence
                        break

            competence_span = None
            text = None
            min_value = None
            max_value = None
            if competence_constraint is not None:
                competence_span = competence_constraint.get("competence_attr_name")
                if competence_span is not None:
                    text = extract_text(competence_span,
                                        eligible_charter["analysis"]["tokenization_maps"]["words"],
                                        eligible_charter["analysis"]["normal_text"]) + "(" + get_nearest_header(eligible_charter["analysis"]["headers"], competence_span[0])["value"] + ")"
                if competence_constraint["min"] != 0:
                    min_value = {"value": competence_constraint["min"], "currency": competence_constraint["currency_min"]}
                if competence_constraint["max"] != np.inf:
                    max_value = {"value": competence_constraint["max"], "currency": competence_constraint["currency_max"]}

            contract_org2_type = None
            contract_org2_name = None
            if contract_attrs.get("orgs") is not None and len(contract_attrs['orgs']) > 1:
                contract_org2_type = contract_attrs["orgs"][1].get('value')
                contract_org2_name = contract_attrs["orgs"][1].get("value")

            if eligible_protocol is not None:
                eligible_protocol_attrs = get_attrs(eligible_protocol)
                protocol_structural_level = None
                if eligible_protocol_attrs.get("org_structural_level") is not None:
                    protocol_structural_level = eligible_protocol_attrs["org_structural_level"]["value"]

                if eligible_protocol_attrs["date"]["value"] > contract_attrs["date"]["value"]:
                    violations.append(create_violation(
                        {"id": contract["_id"], "number": contract_number,
                         "type": contract["documentType"]},
                        {"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]},
                        {"id": eligible_charter["_id"], "attribute": competence_span, "text": text},
                        "contract_date_less_than_protocol_date",
                        {"contract": {"number": contract_number,
                                      "date": contract_attrs["date"]["value"],
                                      "org_type": contract_org2_type,
                                      "org_name": contract_org2_name},
                         "protocol": {"org_structural_level": protocol_structural_level,
                                      "date": eligible_protocol_attrs["date"]["value"]}}))
                else:
                    if protocol_value is not None:
                        if sign < 0 and min_constraint <= protocol_value["value"] < contract_value["value"]:
                            violations.append(create_violation(
                                {"id": contract["_id"], "number": contract_number,
                                 "type": contract["documentType"]},
                                {"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]},
                                {"id": eligible_charter["_id"], "attribute": competence_span, "text": text},
                                "contract_value_great_than_protocol_value",
                                {"contract": {"number": contract_number,
                                              "date": contract_attrs["date"]["value"],
                                              "org_type": contract_org2_type,
                                              "org_name": contract_org2_name,
                                              "value": contract_value["original_value"],
                                              "currency": contract_value["original_currency"]},
                                "protocol": {
                                     "org_structural_level": protocol_structural_level, "date": eligible_protocol_attrs["date"]["value"],
                                     "value": protocol_value["original_value"], "currency": protocol_value["original_currency"]}}))

                        if sign == 0 and min_constraint <= protocol_value["value"] != contract_value["value"]:
                            violations.append(create_violation(
                                {"id": contract["_id"], "number": contract_number,
                                 "type": contract["documentType"]},
                                {"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]},
                                {"id": eligible_charter["_id"], "attribute": competence_span, "text": text},
                                "contract_value_not_equal_protocol_value",
                                {"contract": {"number": contract_number,
                                              "date": contract_attrs["date"]["value"],
                                              "org_type": contract_org2_type,
                                              "org_name": contract_org2_name,
                                              "value": contract_value["original_value"],
                                              "currency": contract_value["original_currency"]},
                                "protocol": {
                                     "org_structural_level": protocol_structural_level, "date": eligible_protocol_attrs["date"]["value"],
                                     "value": protocol_value["original_value"], "currency": protocol_value["original_currency"]}}))

                        if sign > 0 and min_constraint <= protocol_value["value"] > contract_value["value"]:
                            violations.append(create_violation(
                                {"id": contract["_id"], "number": contract_number,
                                 "type": contract["documentType"]},
                                {"id": eligible_charter["_id"],
                                 "date": eligible_charter_attrs["date"]["value"]},
                                {"id": eligible_charter["_id"], "attribute": competence_span, "text": text},
                                "contract_value_less_than_protocol_value",
                                {"contract": {"number": contract_number,
                                              "date": contract_attrs["date"]["value"],
                                              "org_type": contract_org2_type,
                                              "org_name": contract_org2_name,
                                              "value": contract_value["original_value"],
                                              "currency": contract_value["original_currency"]},
                                "protocol": {
                                     "org_structural_level": protocol_structural_level,
                                     "date": eligible_protocol_attrs["date"]["value"],
                                     "value": protocol_value["original_value"],
                                     "currency": protocol_value["original_currency"]}}))

            else:
                if need_protocol_check:
                    violations.append(create_violation(
                        {"id": contract["_id"], "number": contract_number,
                         "type": contract["documentType"]},
                        {"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]},
                        {"id": eligible_charter["_id"], "attribute": competence_span, "text": text},
                        {"type": "protocol_not_found", "subject": contract_attrs["subject"]["value"],
                         "org_structural_level": org_level,
                         "min": min_value,
                         "max": max_value
                         },
                        {"contract": {"number": contract_number,
                                      "date": contract_attrs["date"]["value"],
                                      "org_type": contract_org2_type,
                                      "org_name": contract_org2_name,
                                      "value": contract_value["original_value"],
                                      "currency": contract_value["original_currency"]
                                      }}))
    return violations


def get_amount_netto(price):
    if price is None:
        return None
    result = {}
    # price_obj = json.loads(json.dumps(price), object_hook=lambda item: SimpleNamespace(**item))
    if price.get('currency') is not None:
        result['currency'] = price['currency']['value']
    else:
        return None
    if price.get('amount_netto') is not None:
        result['value'] = price['amount_netto']['value']
        return result
    elif price.get('amount_brutto') is not None:
        vat_unit = price['currency']['value']
        if price.get('vat_unit') is not None:
            vat_unit = price['vat_unit']['value']
        if price.get('vat') is None:
            vat_value = 20
            vat_unit = 'Percent'
        else:
            vat_value = price['vat']['value']
        if vat_unit == 'Percent':
            result['value'] = price['amount_brutto']['value'] * (100 - vat_value) / 100.0
        elif vat_unit != price['currency']['value']:
            vat = convert_to_currency({'value': vat_value, 'currency': vat_unit}, price['currency']['value'])
            result['value'] = price['amount_brutto']['value'] - vat['value']
        else:
            result['value'] = price['amount_brutto']['value'] - vat_value
    elif price.get('amount') is not None:
        result['value'] = price['amount']['value']
    if result.get('value') is None:
        return None
    return result


def check_inside(document, additional_docs):
    doc_attrs = get_attrs(document)
    inside_info = None
    if doc_attrs.get('insideInformation') is not None:
        inside_info = doc_attrs['insideInformation']
    elif doc_attrs.get('subject') is not None and doc_attrs['subject'].get('insideInformation') is not None:
        inside_info = doc_attrs['subject']['insideInformation']
    amount_netto = find_contract_amount_netto(document, additional_docs)
    gpn_book_value = get_latest_gpn_book_value()
    if amount_netto is not None and gpn_book_value is not None:
        if amount_netto['currency'] != 'RUB':
            amount_netto = convert_to_currency(amount_netto, 'RUB')
        if amount_netto['value'] > gpn_book_value['value'] * 0.1:
            return {'type': 'InsiderControl', 'text': 'Крупная сделка(сумма договора более 10% балансовой стоимости ГПН)', 'reason': '', 'notes': [], 'inside_type': 'Deals'}

    if inside_info is not None:
        text = extract_text(inside_info['span'], document["analysis"]["tokenization_maps"]["words"], document["analysis"]["normal_text"])
        return {'type': 'InsiderControl', 'text': text, 'reason': '', 'notes': [], 'inside_type': inside_info['value']}
    return None


def prepare_affiliates(legal_entity_types):
    result = []
    coll = get_mongodb_connection().get_collection('affiliatesList')
    affiliates = coll.find({})
    for affiliate in affiliates:
        company = False
        for key, value in legal_entity_types.items():
            if affiliate['name'].lower().strip().startswith(key):
                affiliate['clean_name'] = affiliate['name'][len(key):].strip().replace('"', '').replace("'", '').replace('«', '').replace('»', '')
                affiliate['legal_entity_type'] = key
                company = True
                break
            if affiliate['name'].lower().strip().startswith(value + ' '):
                affiliate['clean_name'] = affiliate['name'][len(value):].strip().replace('"', '').replace("'", '').replace('«', '').replace('»', '')
                affiliate['legal_entity_type'] = key
                company = True
                break
        if not company:
            affiliate['last_name'] = affiliate['name'].split(' ')[0]
        result.append(affiliate)
    return result


def prepare_beneficiary_chain(audit, legal_entity_types):
    result = []
    if audit.get('beneficiary_chain') is None:
        return result
    for beneficiary in audit['beneficiary_chain']['benefeciaries']:
        if beneficiary.get('name') is not None:
            match = re.search(company_name_pattern, beneficiary['name'])
            if match is not None:
                beneficiary['clean_name'] = normalize_only_company_name(match.group('company_name'))
                without_name = re.sub(company_name_pattern, '', beneficiary['name'])
                for key, value in legal_entity_types.items():
                    if key.lower() in without_name.lower():
                        beneficiary['legal_entity_type'] = key
                    if value and value.lower() in without_name.lower():
                        beneficiary['legal_entity_type'] = key
            else:
                for key, value in legal_entity_types.items():
                    if beneficiary['name'].lower().strip().startswith(key.lower()):
                        beneficiary['clean_name'] = beneficiary['name'][len(key):].strip().replace('"', '').replace("'", '').replace('«', '').replace('»', '')
                        beneficiary['legal_entity_type'] = key
                        beneficiary['clean_name'] = normalize_only_company_name(beneficiary['clean_name'])
                    if beneficiary['name'].lower().strip().startswith(value.lower() + ' '):
                        beneficiary['clean_name'] = beneficiary['name'][len(value):].strip().replace('"', '').replace("'", '').replace('«', '').replace('»', '')
                        beneficiary['clean_name'] = normalize_only_company_name(beneficiary['clean_name'])
                        beneficiary['legal_entity_type'] = key
        if beneficiary.get('namePerson') is not None:
            company = False
            match = re.search(company_name_pattern, beneficiary['namePerson'])
            if match is not None:
                beneficiary['clean_name_person'] = match.group('company_name')
                without_name = re.sub(company_name_pattern, '', beneficiary['namePerson'])
                company = True
                for key, value in legal_entity_types.items():
                    if key.lower() in without_name.lower():
                        beneficiary['legal_entity_type_name_person'] = key
                    if value and value.lower() in without_name.lower():
                        beneficiary['legal_entity_type_name_person'] = key
            else:
                for key, value in legal_entity_types.items():
                    if beneficiary['namePerson'].lower().strip().startswith(key.lower()):
                        beneficiary['clean_name_person'] = beneficiary['namePerson'][len(key):].strip().replace('"', '').replace("'", '').replace('«', '').replace('»', '')
                        company = True
                    if beneficiary['namePerson'].lower().strip().startswith(value.lower() + ' '):
                        beneficiary['clean_name_person'] = beneficiary['namePerson'][len(value):].strip().replace('"', '').replace("'", '').replace('«', '').replace('»', '')
                        company = True
            if not company:
                beneficiary['last_name'] = beneficiary['namePerson'].split(' ')[0]
            else:
                beneficiary['clean_name_person'] = normalize_only_company_name(beneficiary['clean_name_person'])
        result.append(beneficiary)
    return result


def is_same_person(name1, name2):
    match1 = re.search(full_name_pattern, name1)
    match2 = re.search(full_name_pattern, name2)
    if match1 is not None and match2 is not None:
        first_name1 = match1.group('first_name')
        middle_name1 = match1.group('middle_name')
        first_name2 = match2.group('first_name')
        middle_name2 = match2.group('middle_name')
        if textdistance.jaro_winkler.normalized_distance(first_name1, first_name2) < 0.1:
            if middle_name1 is not None and middle_name2 is not None:
                if textdistance.jaro_winkler.normalized_distance(middle_name1, middle_name2) < 0.1:
                    return True
            else:
                return True
    return False


def get_reason(affiliate, contract_date):
    for reason in affiliate['reasons']:
        if reason.get('date') is not None:
            if reason['date'] < contract_date:
                return reason
        else:
            return reason
    return None


def contains_same_name(result, name):
    for elem in result:
        if elem['text'] == name:
            return True
    return False


def get_persons_from_chain(beneficiaries, name):
    result = []
    found_names = [name]
    while True:
        new_names = []
        for beneficiary in beneficiaries:
            for found_name in found_names:
                if beneficiary['name'] == found_name:
                    if beneficiary.get('last_name') is not None:
                        result.append(beneficiary)
                    else:
                        new_names.append(beneficiary['namePerson'])
        found_names = new_names
        if len(found_names) == 0:
            if len(result) == 0:
                return None
            return result
        if name in found_names:
            return None


def find_contract_amount_netto(contract, additional_docs):
    contract_attrs = get_attrs(contract)
    result = get_amount_netto(contract_attrs.get('price'))
    if result is None:
        for additional_doc in additional_docs:
            doc_attrs = get_attrs(additional_doc)
            result = get_amount_netto(doc_attrs.get('price'))
            if result is not None:
                return result
    return result


def is_already_added(result, beneficiary):
    for elem in result:
        if elem['name'] == beneficiary['name'] and elem['namePerson'] == beneficiary['namePerson']:
            return True
    return False


def build_chain(name, beneficiaries):
    result = []
    links = deque()
    participants = []
    for beneficiary in beneficiaries:
        if beneficiary.get('clean_name') == name and not is_already_added(result, beneficiary):
            if (beneficiary.get('last_name') is None and beneficiary.get('share') is not None and beneficiary['share'] > 50) or beneficiary.get('last_name') is not None:
                links.append(beneficiary)
                result.append(beneficiary)
            if beneficiary.get('share') is None and ('участник' in beneficiary['roles'] or 'акционер' in beneficiary['roles']):
                participants.append(beneficiary)
    if len(participants) == 1 and not is_already_added(result, participants[0]):
        links.append(participants[0])
        result.append(participants[0])

    while len(links) > 0:
        beneficiary_link = links.popleft()
        participants = []
        for beneficiary in beneficiaries:
            if beneficiary_link['namePerson'] == beneficiary['name'] and not is_already_added(result, beneficiary):
                beneficiary['parent'] = beneficiary_link
                if (beneficiary.get('last_name') is None and beneficiary.get('share') is not None and beneficiary['share'] > 50) or beneficiary.get('last_name') is not None:
                    links.append(beneficiary)
                    result.append(beneficiary)
                if beneficiary.get('share') is None and ('участник' in beneficiary['roles'] or 'акционер' in beneficiary['roles']):
                    participants.append(beneficiary)
        if len(participants) == 1 and not is_already_added(result, participants[0]):
            links.append(participants[0])
            result.append(participants[0])
    return result


def find_org_interest(result, name, interests):
    for key, value in list(reversed(sorted(interests.items()))):
        if value is not None:
            for org in value['organizations']:
                if compare_ignore_case(org.get('clean_name'), name) or compare_ignore_case(org.get('clean_short_name'), name):
                    share = 100
                    if len(org['shareChain']) > 0:
                        share = org["shareChain"][-1]['percents']
                    control_type = 'прямой контроль'
                    if org.get('control_type') is not None:
                        control_type = org['control_type']
                    control_chain = ''
                    for controlled_org in org['shareChain']:
                        control_chain += '<br>' + controlled_org['text']
                    reason_text = f'Контролируемая доля {companies.get(key)} в уставном капитале контрагента {share}%<br>{control_type} эмитента: {control_chain}'
                    result.append({'type': 'InterestControl', 'text': companies.get(key), 'reason': reason_text, 'notes': []})
                    return True
    return False


def find_person_interest(result, beneficiary, interests):
    last_name = beneficiary['last_name']
    full_name = None
    reason_text = ''
    notes = []
    for key, value in list(reversed(sorted(interests.items()))):
        if value is not None:
            for person in value['stakeholders']:
                if textdistance.jaro_winkler.normalized_distance(last_name, person['last_name']) < 0.1:
                    if full_name is None and is_same_person(beneficiary['namePerson'], person['name']):
                        full_name = person['name']
                        gp_roles = ''
                        contragent_roles = ','.join(beneficiary['roles'])
                        for reason in person['reasons']:
                            if reason.get('endDate') is None:
                                gp_roles += '<br>' + reason['organization'] + ', ' + reason['text']
                        reason_text = f'Должности занимаемые в структурах ГП и ГПН: {gp_roles} <br>Должности в структуре контрагента: {contragent_roles}'
                    else:
                        notes.append(person['name'])
    if full_name is not None or len(notes) > 0:
        result.append({'type': 'InterestControl', 'text': full_name, 'reason': reason_text, 'notes': notes})
        return True
    return False


def find_gp_gpn(result, chain):
    company_names = companies.values()
    for beneficiary in chain:
        if beneficiary.get('clean_name_person') is not None:
            for company_name in company_names:
                if compare_ignore_case(beneficiary['clean_name_person'], company_name):
                    share = 100
                    control_type = 'прямой контроль'
                    control_chain = ''
                    count = 0
                    org = beneficiary
                    while org is not None:
                        if count > 0:
                            control_type = 'косвенный контроль'
                        org_share = 100
                        if org.get('share') is not None:
                            org_share = org['share']
                        name = org["name"]
                        if org.get('clean_name') is not None and org.get('legal_entity_type') is not None:
                            name = org['legal_entity_type'] + ' ' + org['clean_name']
                        name_person = org["namePerson"]
                        if org.get('clean_name_person') is not None and org.get('legal_entity_type_name_person') is not None:
                            name_person = org['legal_entity_type_name_person'] + ' ' + org['clean_name_person']
                        control_chain += f'<br>доля {name_person} в {name} - {org_share}%'
                        share = org_share
                        org = org.get('parent')
                        count += 1
                    name_person = beneficiary["namePerson"]
                    if beneficiary.get('clean_name_person') is not None and beneficiary.get('legal_entity_type_name_person') is not None:
                        name_person = beneficiary['legal_entity_type_name_person'] + ' ' + beneficiary['clean_name_person']
                    reason_text = f'Контролируемая доля {name_person} в уставном капитале контрагента {share}%<br>{control_type} эмитента: {control_chain}'
                    result.append({'type': 'InterestControl', 'text': name_person, 'reason': reason_text, 'notes': []})
                    return True
    return False


def check_interest(contract, additional_docs, interests, beneficiaries):
    result = []
    contract_attrs = get_attrs(contract)

    amount_netto = find_contract_amount_netto(contract, additional_docs)
    if amount_netto is not None:
        if amount_netto['currency'] != 'RUB':
            amount_netto = convert_to_currency(amount_netto, 'RUB')
        if amount_netto['value'] >= 1000000000:#need interest check
            if contract_attrs.get('orgs') is not None:
                for i, org in enumerate(contract_attrs['orgs']):
                    if i != 0 and org.get('name') is not None:
                        org_name = org['name']['value'].strip().replace('"', '').replace("'", '').replace('«', '').replace('»', '')
                        org_name = normalize_only_company_name(org_name)
                        find_org_interest(result, org_name, interests)
                        chain = build_chain(org_name, beneficiaries)
                        if len(chain) == 0 and org.get('alt_name') is not None:
                            org_name = org['alt_name']['value'].strip().replace('"', '').replace("'", '').replace('«', '').replace('»', '')
                            org_name = normalize_only_company_name(org_name)
                            find_org_interest(result, org_name, interests)
                            chain = build_chain(org_name, beneficiaries)

                        find_gp_gpn(result, chain)
                        for beneficiary in chain:
                            if beneficiary.get('last_name') is not None:
                                find_person_interest(result, beneficiary, interests)
                            else:
                                find_org_interest(result, beneficiary['clean_name_person'], interests)
    return result


def check_contract_project(document, audit, interests, beneficiaries, docs):
    violations = []
    document_attrs = get_attrs(document)
    if document.get('documentType') in ['CONTRACT', 'AGREEMENT', 'SUPPLEMENTARY_AGREEMENT']:
        additional_docs = list(filter(lambda x: x['_id'] != document['_id'], docs))
        if 'InterestControl' in audit['checkTypes']:
            interest_violations = check_interest(document, additional_docs, interests, beneficiaries)
            violations.extend(interest_violations)

        if 'InsiderControl' in audit['checkTypes']:
            violation = check_inside(document, additional_docs)
            if violation is not None:
                violations.append(violation)
    if len(violations) > 0:
        orgs = []
        if document_attrs.get('orgs') is not None and len(document_attrs['orgs']) > 1:
            orgs = document_attrs['orgs'][1:]
        return {'document_id': document['_id'], 'document_filename': document['filename'], 'orgs': orgs, 'violations': violations, 'userViolation': False, 'id': ObjectId()}
    else:
        return None


def exclude_same_charters(charters):
    result = []
    date_map = {}
    for charter in charters:
        if get_attrs(charter).get("date") is not None:
            charter_date = get_attrs(charter)["date"]["value"]
            same_charters = date_map.get(charter_date)
            if same_charters is None:
                date_map[charter_date] = [charter]
            else:
                same_charters.append(charter)

    for date, same_charters in date_map.items():
        if len(same_charters) == 1:
            result.append(same_charters[0])
        else:
            best = same_charters[0]
            for charter in same_charters[1:]:
                if charter.get("user") is not None:
                    if best.get("user") is not None:
                        if charter["user"]["updateDate"] > best["user"]["updateDate"]:
                            best = charter
                    else:
                        best = charter
                else:
                    if best.get("user") is None and best["analysis"]["analyze_timestamp"] > charter["analysis"]["analyze_timestamp"]:
                        best = charter
            result.append(best)
    return result


def prepare_interests(interest):
    if interest is not None:
        for org in interest['organizations']:
            for key, value in legal_entity_types.items():
                if org.get('name') is not None and org['name'].lower().strip().startswith(key.lower()):
                    org['clean_name'] = normalize_only_company_name(org['name'][len(key):].strip().replace('"', '').replace("'", '').replace('«', '').replace('»', ''))
                    org['legal_entity_type'] = key
                if org.get('shortName') is not None and org['shortName'].lower().strip().startswith(value.lower() + ' '):
                    org['clean_short_name'] = normalize_only_company_name(org['shortName'][len(value):].strip().replace('"', '').replace("'", '').replace('«', '').replace('»', ''))
        for person in interest['stakeholders']:
            person['last_name'] = person['name'].split(' ')[0]


def get_latest_interest():
    result = {}
    db = get_mongodb_connection()
    reports = db['quarterlyReport']
    result['gp'] = reports.find_one({'company': 'gp'}, sort=[('uploadDate', pymongo.DESCENDING)])
    prepare_interests(result.get('gp'))
    result['gpn'] = reports.find_one({'company': 'gpn'}, sort=[('uploadDate', pymongo.DESCENDING)])
    prepare_interests(result.get('gpn'))
    return result


def get_latest_gpn_book_value():
    db = get_mongodb_connection()
    coll = db['bookvalues']
    return coll.find_one({}, sort=[('date', pymongo.DESCENDING)])


def save_email_sending_result(result, audit):
    db = get_mongodb_connection()
    db["audits"].update_one({'_id': audit["_id"]}, {"$set": {"mail_sent": result}})


def finalize():
    audits = get_audits()
    interests = None
    for audit in audits:
        if audit.get('pre-check'):
            logger.info(f'.....finalizing pre-audit {audit["_id"]}')
            prepared_beneficiaries = None
            if 'InterestControl' in audit['checkTypes']:
                if interests is None:
                    interests = get_latest_interest()
                prepared_beneficiaries = prepare_beneficiary_chain(audit, legal_entity_types)
            documents = get_docs_by_audit_id(audit["_id"], 15, without_large_fields=True)
            violations = []
            for document_id in documents:
                try:
                    document = get_doc_by_id(document_id["_id"])
                    violation = check_contract_project(document, audit, interests, prepared_beneficiaries, documents)
                    if violation is not None:
                        violations.append(violation)
                except Exception as err:
                    logger.exception(f'cant finalize document {document_id["_id"]}')

            save_violations(audit, violations)
            logger.info(f'.....pre-audit {audit["_id"]} is waiting for approval')
            continue
        if "Все ДО" in audit["subsidiary"]["name"]:
            logger.info(f'.....audit {audit["_id"]} finalizing skipped')
            continue
        logger.info(f'.....finalizing audit {audit["_id"]}')
        violations = []
        contract_ids = get_docs_by_audit_id(audit["_id"], 15, "CONTRACT", id_only=True)
        charters = []
        if audit.get("charters") is not None:
            for charter_id in audit["charters"]:
                charter = get_doc_by_id(charter_id)
                if (charter.get("isActive") is None or charter["isActive"]) and charter["state"] == 15:
                    charters.append(charter)
            cleaned_charters = exclude_same_charters(charters)
            charters = sorted(cleaned_charters, key=lambda k: get_attrs(k)["date"]["value"], reverse=True)
        protocols = get_docs_by_audit_id(audit["_id"], 15, "PROTOCOL", without_large_fields=True)
        supplementary_agreements = get_docs_by_audit_id(audit["_id"], 15, "SUPPLEMENTARY_AGREEMENT", without_large_fields=True)

        for document_id in contract_ids:
            try:
                contract = get_doc_by_id(document_id["_id"])
                if get_attrs(contract).get('date') is None:
                    continue
                violations.extend(check_contract(contract, charters, protocols, audit, supplementary_agreements))
            except Exception as err:
                logger.exception(f'cant finalize contract {document_id["_id"]}')

        save_violations(audit, violations)
        logger.info(f'.....audit {audit["_id"]} is waiting for approval')
        if audit.get('mail_sent') is None or not audit['mail_sent']:
            result = mail.send_end_audit_email(audit)
            save_email_sending_result(result, audit)


if __name__ == '__main__':
    finalize()

