from datetime import datetime

from zeep import Client, helpers

import gpn_config
from analyser.finalizer import normalize_only_company_name
from analyser.log import logger
from analyser.structures import legal_entity_types
from gpn.gpn import update_subsidiaries_cache
from gpn_config import secret
from integration.db import get_mongodb_connection

_client = None



def get_csgk_client():
    try:
        global _client
        wsdl = gpn_config.configured('GPN_CSGK_WSDL')
        if _client is None:
            if wsdl is not None:
                logger.info(f"CSGK WSDL: {wsdl}")
                _client = Client(wsdl=wsdl)
        return _client
    except Exception as e:
        logger.exception(e)
    return None


def _clean_short_subsidiary_name(short_name, short_legal_entities):
    split = short_name.split(' ', 1)
    if len(split) > 1:
        for legal_entity in short_legal_entities:
            if legal_entity == split[0].strip():
                name = normalize_only_company_name(split[1].strip().replace('"', '').replace("'", ''))
                return legal_entity, name
    name = normalize_only_company_name(short_name)
    return '', name


def _clean_subsidiary_name(name, legal_entities):
    for legal_entity in legal_entities:
        if name.lower().startswith(legal_entity):
            result = normalize_only_company_name(name[len(legal_entity):].strip().replace('"', '').replace("'", ''))
            return result
    result = normalize_only_company_name(name)
    return result


def get_subsidiary_list():
    try:
        client = get_csgk_client()
        user = secret('GPN_CSGK_USER')
        password = secret('GPN_CSGK_PASSWORD')
        if client is not None and user is not None and password is not None:
            subsidiaries = {}
            result_raw = client.service.Execute(user, password, 'Get_CompanySimple_List')
            result = helpers.serialize_object(result_raw)

            ret_code = result.get('RetCode')
            ret_msg = result.get('RetMsg')
            if ret_code == 0:
                legal_entity_aliases = list(filter(lambda x: len(x) > 0, legal_entity_types.values()))
                lower_legal_entities = list(map(lambda x: x.lower(), legal_entity_types.keys()))
                for sub_result in result['ExecuteResult']['_value_1']['_value_1']:
                    csgk_sub = sub_result.get('CorpManagement._x0020_Integration_CompanySimple_List')
                    clean_name = _clean_subsidiary_name(csgk_sub.get('ULName'), lower_legal_entities)
                    legal_entity_type, clean_short_name = _clean_short_subsidiary_name(csgk_sub.get('ULShortName'), legal_entity_aliases)
                    subsidiary = {
                        'subsidiary_id': csgk_sub.get('IdCompany'),
                        '_id': clean_name,
                        'legal_entity_type': legal_entity_type,
                        'aliases': [clean_short_name],
                        'short_name': csgk_sub.get('ULShortName'),
                        'INN': csgk_sub.get('INN'),
                        'KPP': csgk_sub.get('KPP'),
                        'OGRN': csgk_sub.get('OGRN'),
                        'CuratorDO_Name': csgk_sub.get('CuratorDO_Name'),
                        'CuratorDO_Email': csgk_sub.get('CuratorDO_Email')
                    }
                    subsidiaries[subsidiary['_id']] = subsidiary
                return list(subsidiaries.values())
            else:
                logger.error(f'CSGK returned {ret_code} code. Error message: {ret_msg}')
    except Exception as e:
        logger.exception(e)
    return None


def get_shareholders():
    try:
        client = get_csgk_client()
        user = secret('GPN_CSGK_USER')
        password = secret('GPN_CSGK_PASSWORD')
        if client is not None and user is not None and password is not None:
            shareholders = []
            result_raw = client.service.Execute(user, password, 'Get_CompanyShareHolder_List')
            result = helpers.serialize_object(result_raw)

            ret_code = result.get('RetCode')
            ret_msg = result.get('RetMsg')
            if ret_code == 0:
                for sub_result in result['ExecuteResult']['_value_1']['_value_1']:
                    csgk_sub = sub_result.get('Integration._x0020_Get_CompanyShareHolder_List_V')
                    shareholder = {
                        'TypeShareHolder': csgk_sub.get('TypeShareHolder'),
                        'name': csgk_sub.get('ShareHolderName'),
                        'share': float(csgk_sub.get('ShareFraction')),
                        'reasons': [{'text': 'Акционер'}],
                        'shortName': csgk_sub.get('ULShortName'),
                        'TypeSH': csgk_sub.get('TypeSH'),
                        'company': 'gpn'
                    }
                    shareholders.append(shareholder)
                return shareholders
            else:
                logger.error(f'CSGK returned {ret_code} code. Error message: {ret_msg}')
    except Exception as e:
        logger.exception(e)
    return None


def get_board_of_directors():
    try:
        client = get_csgk_client()
        user = secret('GPN_CSGK_USER')
        password = secret('GPN_CSGK_PASSWORD')
        if client is not None and user is not None and password is not None:
            shareholders = []
            result_raw = client.service.Execute(user, password, 'Get_CompanyAuthoritySD_List')
            result = helpers.serialize_object(result_raw)

            ret_code = result.get('RetCode')
            ret_msg = result.get('RetMsg')
            if ret_code == 0:
                for sub_result in result['ExecuteResult']['_value_1']['_value_1']:
                    csgk_sub = sub_result.get('Integration._x0020_Get_StockCompanyAuthoritySD_List_V')
                    shareholder = {
                        'name': csgk_sub.get('AuthorityMemberName'),
                        'gpn_employee': bool(csgk_sub.get('IsEmployeeGPN')),
                        'reasons': [{'text': csgk_sub.get('AuthorityMemberRoleName')}],
                        'shortName': csgk_sub.get('ULShortName'),
                        'company': 'gpn'
                    }
                    shareholders.append(shareholder)
                return shareholders
            else:
                logger.error(f'CSGK returned {ret_code} code. Error message: {ret_msg}')
    except Exception as e:
        logger.exception(e)
    return None


def sync_csgk_data():
    if gpn_config.configured("GPN_CSGK_WSDL") is None:
        return
    logger.info('Start CSGK synchronization.')
    subsidiaries = get_subsidiary_list()
    db = get_mongodb_connection()
    if subsidiaries is not None:
        update_subsidiaries_cache(subsidiaries)
        coll = db["subsidiaries"]
        coll.delete_many({})
        coll.insert_many(subsidiaries)

    # stakeholders = []
    # shareholders = get_shareholders()
    # if shareholders is not None:
    #     stakeholders.extend(shareholders)
    # board_of_directors = get_board_of_directors()
    # if board_of_directors is not None:
    #     stakeholders.extend(board_of_directors)
    # if len(stakeholders) > 0:
    #     coll = db['affiliatesList']
    #     coll.delete_many({'company': 'gpn'})
    #     coll.insert_many(stakeholders)
    db['catalog'].insert({'last_csgk_sync_date': datetime.today()})
    logger.info('CSGK synchronization finished.')

