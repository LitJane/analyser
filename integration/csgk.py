import os
from datetime import datetime

from zeep import Client, helpers

from analyser.log import logger
from gpn.gpn import update_subsidiaries_cache
from integration.db import get_mongodb_connection

_client = None


def _env_var(vname, default_val=None):
    if vname not in os.environ:
        msg = f'CSGK : define {vname} environment variable! defaulting to {default_val}'
        logger.warning(msg)
        return default_val
    else:
        return os.environ[vname]


def get_csgk_client():
    try:
        global _client
        wsdl = _env_var('GPN_CSGK_WSDL')
        if _client is None:
            if wsdl is not None:
                logger.info(f"CSGK WSDL: {wsdl}")
                _client = Client(wsdl=wsdl)
        return _client
    except Exception as e:
        logger.exception(e)
    return None


def _get_legal_entity_type(short_name, legal_entities):
    candidate = short_name.split(' ', 1)[0].strip()
    for legal_entity in legal_entities:
        if legal_entity == candidate:
            return legal_entity
    return ''


def get_subsidiary_list():
    try:
        client = get_csgk_client()
        user = _env_var('GPN_CSGK_USER')
        password = _env_var('GPN_CSGK_PASSWORD')
        if client is not None and user is not None and password is not None:
            subsidiaries = []
            result_raw = client.service.Execute(user, password, 'Get_CompanySimple_List')
            result = helpers.serialize_object(result_raw)

            ret_code = result.get('RetCode')
            ret_msg = result.get('RetMsg')
            if ret_code == 0:
                legal_entity_aliases = []
                legal_entity_aliases = list(filter(lambda x: len(x) > 0, legal_entity_aliases))
                for sub_result in result['ExecuteResult']['_value_1']['_value_1']:
                    csgk_sub = sub_result.get('CorpManagement._x0020_Integration_CompanySimple_List')
                    subsidiary = {
                        'subsidiary_id': csgk_sub.get('IdCompany'),
                        '_id': csgk_sub.get('ULName'),
                        'legal_entity_type': _get_legal_entity_type(csgk_sub.get('ULShortName'), legal_entity_aliases),
                        'aliases': [csgk_sub.get('ULShortName')],
                        'short_name': csgk_sub.get('ULShortName'),
                        'INN': csgk_sub.get('INN'),
                        'KPP': csgk_sub.get('KPP'),
                        'OGRN': csgk_sub.get('OGRN'),
                        'CuratorDO_Name': csgk_sub.get('CuratorDO_Name'),
                        'CuratorDO_Email': csgk_sub.get('CuratorDO_Email')
                    }
                    subsidiaries.append(subsidiary)
                return subsidiaries
            else:
                logger.error(f'CSGK returned {ret_code} code. Error message: {ret_msg}')
    except Exception as e:
        logger.exception(e)
    return None


def get_shareholders():
    try:
        client = get_csgk_client()
        user = _env_var('GPN_CSGK_USER')
        password = _env_var('GPN_CSGK_PASSWORD')
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
        user = _env_var('GPN_CSGK_USER')
        password = _env_var('GPN_CSGK_PASSWORD')
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
    if os.environ.get("GPN_CSGK_WSDL") is None:
        return
    logger.info('Start CSGK synchronization.')
    subsidiaries = get_subsidiary_list()
    db = get_mongodb_connection()
    if subsidiaries is not None:
        update_subsidiaries_cache(subsidiaries)
        coll = db["subsidiaries"]
        coll.delete_many({})
        coll.insert_many(subsidiaries)

    stakeholders = []
    shareholders = get_shareholders()
    if shareholders is not None:
        stakeholders.extend(shareholders)
    board_of_directors = get_board_of_directors()
    if board_of_directors is not None:
        stakeholders.extend(board_of_directors)
    if len(stakeholders) > 0:
        coll = db['affiliatesList']
        coll.delete_many({'company': 'gpn'})
        coll.insert_many(stakeholders)
    db['catalog'].insert({'last_csgk_sync_date': datetime.today()})
    logger.info('CSGK synchronization finished.')

