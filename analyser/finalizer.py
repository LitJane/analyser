import json
import re
from types import SimpleNamespace

import numpy as np
import pymongo
import textdistance
from bson import ObjectId

from analyser.log import logger
from integration.currencies import convert_to_currency
from integration.db import get_mongodb_connection


full_name_pattern = re.compile(r'(?P<last_name>[а-я,А-Я,a-z,A-Z]+) +(?P<first_name>[а-я,А-Я,a-z,A-Z]+)(\.? +)?(?P<middle_name>[а-я,А-Я,a-z,A-Z]+)?')


def get_audits():
    db = get_mongodb_connection()
    audits_collection = db['audits']

    res = audits_collection.find({'status': 'Finalizing'}).sort([("createDate", pymongo.ASCENDING)])
    return res


def remove_old_links(audit_id, contract_id):
    db = get_mongodb_connection()
    audit_collection = db['audits']
    audit_collection.update_one({"_id": audit_id}, {"$pull": {"links": {"type": "analysis", "$or": [{"toId": contract_id}, {"fromId": contract_id}]}}})


def get_linked_docs(audit_id, contract_id):
    db = get_mongodb_connection()
    audit_collection = db['audits']
    links = audit_collection.find({"_id": audit_id, "$or": [{"links.toId": contract_id}, {"links.fromId": contract_id}]})
    result = []
    document_collection = db['documents']
    for link in links:
        if link['fromId'] == contract_id:
            result.append(document_collection.find_one({'_id': link['toId']}))
        else:
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
    last_idx = words[span[1]][0] - 1
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


def get_max_value(doc_attrs):
    max_value = None
    sign = 0
    for key, value in doc_attrs.items():
        if key.endswith("/value"):
            if doc_attrs.get(key[:-5] + "sign") is not None:
                sign = doc_attrs[key[:-5] + "sign"]["value"]
            current_value = convert_to_currency({"value": value["value"], "currency": doc_attrs[key[:-5] + "currency"]["value"]})
            if max_value is None or max_value["value"] < current_value["value"]:
                max_value = current_value
    return max_value, sign


def get_charter_diapasons(charter):
    #group by subjects
    subjects = {}
    charter_attrs = get_attrs(charter)
    min_constraint = np.inf
    charter_currency = 'RUB'
    if charter_attrs.get('structural_levels') is not None:
        for structural_level in charter_attrs['structural_levels']:
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


def find_protocol(contract, protocols, org_level, audit):
    contract_attrs = get_attrs(contract)
    result = []
    for protocol in protocols:
        protocol_attrs = get_attrs(protocol)
        if protocol_attrs.get("structural_level") is not None and protocol_attrs["structural_level"]["value"] == org_level and protocol_attrs.get('agenda_items') is not None:
            for agenda_item in protocol_attrs['agenda_items']:
                if agenda_item.get('contracts') is not None:
                    for agenda_contract in agenda_item['contracts']:
                        if agenda_contract.get('orgs') is not None and contract_attrs.get('orgs') is not None:
                            clean_contract_orgs=[]
                            for contract_org in contract_attrs['orgs']:
                                if contract_org.get('name') is not None and contract_org['name'].get('value') is not None:
                                    clean_contract_orgs.append(clean_name(contract_org['name']['value']))
                            for agenda_org in agenda_contract['orgs']:
                                if agenda_org.get('name') is not None and agenda_org['name'].get('value') is not None:
                                    clean_protocol_org = clean_name(agenda_org['name']["value"])
                                    for clean_contract_org in clean_contract_orgs:
                                        distance = textdistance.levenshtein.normalized_distance(clean_contract_org, clean_protocol_org)
                                        if distance < 0.1:
                                            result.append(protocol)
    if len(result) == 0:
        return None
    else:
        return result[0]


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
    # user_linked_docs = get_linked_docs(audit["_id"], contract["_id"])
    if contract_attrs.get("number") is not None:
        contract_number = contract_attrs["number"]["value"]
        # linked_sup_agreements = list(filter(lambda doc: doc['parse']['documentType'] == 'SUPPLEMENTARY_AGREEMENT', user_linked_docs))
        find_supplementary_agreements(contract, supplementary_agreements, audit)
    eligible_charter = None
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
        if contract_attrs.get('price') is not None:
            contract_value = convert_to_currency({"value": contract_attrs['price']['amount']["value"], "currency": contract_attrs['price']['currency']["value"]}, charter_currency)

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

            for competence, constraint in competences.items():
                if constraint['currency_min'] == 'Percent' and book_value is not None:
                    abs_min = constraint['min'] * book_value / 100
                else:
                    abs_min = constraint['min']
                if constraint['currency_max'] == 'Percent' and book_value is not None:
                    abs_max = constraint['max'] * book_value / 100
                else:
                    abs_max = constraint['max']

                if abs_min <= contract_value["value"] <= abs_max:
                    need_protocol_check = True
                    competence_constraint = constraint
                    eligible_protocol = find_protocol(contract, protocols, competence, audit)
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
                add_link(audit["_id"], contract["_id"], eligible_protocol["_id"])
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
                    protocol_value, sign = get_max_value(eligible_protocol_attrs)
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
                                              "value": contract_attrs["sign_value_currency/value"]["value"],
                                              "currency": contract_attrs["sign_value_currency/currency"]["value"]},
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
                                              "value": contract_attrs["sign_value_currency/value"]["value"],
                                              "currency": contract_attrs["sign_value_currency/currency"]["value"]},
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
                                              "value": contract_attrs["sign_value_currency/value"]["value"],
                                              "currency": contract_attrs["sign_value_currency/currency"][
                                                  "value"]},
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
                                      "value": contract_attrs["sign_value_currency/value"]["value"],
                                      "currency": contract_attrs["sign_value_currency/currency"]["value"]}}))
    return violations


def get_amount_netto(price):
    if price is None:
        return None
    result = {}
    # price_obj = json.loads(json.dumps(price), object_hook=lambda item: SimpleNamespace(**item))
    if price.get('currency') is not None:
        result['currency'] = price['currency']['value']
    if price.get('amount_netto') is not None:
        result['value'] = price['amount_netto']['value']
        return result
    elif price.get('amount_brutto') is not None and price.get('vat') is not None and price.get('vat_unit') is not None:
        if price['vat_unit']['value'] == 'Percent':
            result['value'] = price['amount_brutto']['value'] * (100 - price['vat']['value']) / 100.0
        elif price['vat_unit']['value'] != price['currency']['value']:
            vat = convert_to_currency({'value': price['vat']['value'], 'currency': price['vat_unit']['value']}, price['currency']['value'])
            result['value'] = price['amount_brutto']['value'] - vat
        else:
            result['value'] = price['amount_brutto']['value'] - price['vat']['value']
    elif price.get('amount') is not None:
        result['value'] = price['amount']['value']
    return result


def check_inside(document):
    doc_attrs = get_attrs(document)
    if doc_attrs.get('insideInformation') is not None:
        text = extract_text(doc_attrs['insideInformation']['span'], document["analysis"]["tokenization_maps"]["words"], document["analysis"]["normal_text"])
        return {'type': 'InsiderControl', 'text': text, 'reason': '', 'notes': [], 'inside_type': doc_attrs['insideInformation']['value']}
    return None


def prepare_affiliates(legal_entity_types):
    result = []
    coll = get_mongodb_connection().get_collection('affiliatesList')
    affiliates = coll.find({})
    for affiliate in affiliates:
        exclude = False
        for legal_entity_type in legal_entity_types:
            if legal_entity_type['_id'] in affiliate['name']:
                exclude = True
                break
        if not exclude:
            affiliate['last_name'] = affiliate['name'].split(' ')[0]
            result.append(affiliate)
    return result


def prepare_beneficiary_chain(audit, legal_entity_types):
    result = []
    if audit['beneficiary_chain'] is None:
        return result
    for beneficiary in audit['beneficiary_chain']['benefeciaries']:
        exclude = False
        for legal_entity_type in legal_entity_types:
            if legal_entity_type['id'] in beneficiary['namePerson']:
                exclude = True
                break
        if not exclude:
            beneficiary['last_name'] = beneficiary['namePerson'].split(' ')[0]
            result.append(beneficiary)

    return result


def is_same_person(name1 , name2):
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


def check_interest(contract, audit, affiliates, beneficiaries):
    result = []
    contract_attrs = get_attrs(contract)

    contract_date = None
    if contract_attrs.get('date') is not None:
        contract_date = contract_attrs['date'].get('value')
    amount_netto = get_amount_netto(contract_attrs.get('price'))
    if amount_netto['currency'] != 'RUB':
        amount_netto = convert_to_currency(amount_netto, 'RUB')
    if amount_netto['value'] >= 1000000000:#need interest check
        if contract_attrs.get('people') is not None:
            for i, person in enumerate(contract_attrs['people']):
                person_last_name = person.get('lastName')
                notes = []
                name = None
                reason = None
                if i == 0:
                    if person.get('value') is not None and person_last_name is not None:
                        for beneficiary in beneficiaries:
                            if textdistance.jaro_winkler.normalized_distance(person_last_name['value'], beneficiary['last_name']) < 0.1:
                                if is_same_person(person.get('value'), beneficiary['namePerson']) and name is None:
                                    name = beneficiary['namePerson']
                                else:
                                    notes.append(beneficiary['namePerson'])
                        if name is not None or len(notes) > 0:
                            result.append({'type': 'InterestControl', 'text': name, 'reason': 'Бенефициар', 'notes': notes})
                else:
                    for affiliate in affiliates:
                        if textdistance.jaro_winkler.normalized_distance(person_last_name['value'], affiliate['last_name']) < 0.1:
                            if is_same_person(person.get('value'), affiliate['namePerson']) and name is None:
                                r = get_reason(affiliate, contract_date)
                                if r is not None:
                                    name = affiliate['namePerson']
                                    reason = reason['text']
                            else:
                                notes.append(affiliate['namePerson'])
                        if name is not None or len(notes) > 0:
                            result.append({'type': 'InterestControl', 'text': name, 'reason': reason, 'notes': notes})
    return result


def check_contract_project(document, audit, affiliates, beneficiaries):
    violations = []
    document_attrs = get_attrs(document)
    if document.get('documentType') == 'CONTRACT' and 'InterestControl' in audit['checkTypes']:
        interest_violations = check_interest(document, audit, affiliates, beneficiaries)
        violations.extend(interest_violations)

    if 'InsiderControl' in audit['checkTypes']:
        violation = check_inside(document)
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


def finalize():
    audits = get_audits()
    for audit in audits:
        if audit.get('pre-check'):
            logger.info(f'.....finalizing pre-audit {audit["_id"]}')
            prepared_affiliates = None
            prepared_beneficiaries = None
            if 'InterestControl' in audit['checkTypes']:
                legal_entity_types = get_mongodb_connection()['legalEntityTypes'].find({})
                if prepared_affiliates is None:
                    prepared_affiliates = prepare_affiliates(legal_entity_types)
                prepared_beneficiaries = prepare_beneficiary_chain(audit, legal_entity_types)
            document_ids = get_docs_by_audit_id(audit["_id"], 15, id_only=True)
            violations = []
            for document_id in document_ids:
                try:
                    document = get_doc_by_id(document_id["_id"])
                    violation = check_contract_project(document, audit, prepared_affiliates, prepared_beneficiaries)
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


if __name__ == '__main__':
    finalize()

