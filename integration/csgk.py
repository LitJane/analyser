import os

from zeep import Client

from analyser.log import logger
from gpn.gpn import update_subsidiaries_cache
from integration.db import get_mongodb_connection


def _env_var(vname, default_val=None):
    if vname not in os.environ:
        msg = f'CSGK : define {vname} environment variable! defaulting to {default_val}'
        logger.warning(msg)
        return default_val
    else:
        return os.environ[vname]


def get_csgk_client():
    try:
        wsdl = _env_var('GPN_CSGK_WSDL')
        if wsdl is not None:
            logger.info(f"CSGK WSDL: {wsdl}")
            return Client(wsdl=wsdl)
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
            result = client.service.Execute(user, password, 'Get_CompanySimple_List')
            ret_code = result.get('RetCode')
            ret_msg = result.get('RetMsg')
            if ret_code == 0:
                legal_entity_aliases = []
                legal_entity_aliases = list(filter(lambda x: len(x) > 0, legal_entity_aliases))
                for sub_result in result['_value_1']['_value_1']:
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


def sync_csgk_data():
    if os.environ.get("GPN_CSGK_WSDL") is None:
        return
    subsidiaries = get_subsidiary_list()
    db = get_mongodb_connection()
    if subsidiaries is not None:
        update_subsidiaries_cache(subsidiaries)
        coll = db["subsidiaries"]
        coll.delete_many({})
        coll.insert_many(subsidiaries)


