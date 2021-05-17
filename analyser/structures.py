#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from enum import Enum, unique, EnumMeta

import numpy as np
from keras.utils import to_categorical

legal_entity_types = {
  'Акционерное общество': 'АО',
  'Публичное акционерное общество': 'ПАО',
  'Общество с ограниченной ответственностью': 'ООО',
  'Иностранное общество с ограниченной ответственностью': 'ИООО',
  'Товарищество с ограниченной ответственностью': 'ТОО',
  'Закрытое акционерное общество': 'ЗАО',
  'Открытое акционерное общество': 'ОАО',
  'Государственное автономное учреждение': 'ГАУ',
  'Частное образовательное учреждение': 'ЧОУ',
  'Некоммерческое партнёрство': 'НП',

  'Федеральное государственное унитарное предприятие': 'ФГУП',
  'Федеральное государственное бюджетное образовательное учреждение высшего образования': 'ФГБОУ',
  'Федеральное государственное бюджетное учреждение': 'ФГБУ',
  'Государственное унитарное предприятие': 'ГУП',

  'Муниципальное бюджетное учреждение': 'МБУ',
  'Муниципальное бюджетное образовательное учреждение': 'МБОУ',
  'Государственное бюджетное образовательное учреждение': 'ГБУ',
  'Государственное бюджетное учреждение': 'МБОУ',

  'Благотворительный фонд': '',
  # 'Фонд':'',
  'Индивидуальный предприниматель': 'ИП',
  'Автономная некоммерческая организация': 'АНО',
}


class DisplayStringEnumMeta(EnumMeta):
  def __new__(mcs, name, bases, attrs):
    obj = super().__new__(mcs, name, bases, attrs)
    obj._value2member_map_ = {}
    for m in obj:
      value, display_string = m.value
      m._value_ = value
      m.display_string = display_string
      obj._value2member_map_[value] = m

    return obj


@unique
class DocumentState(Enum):
  Unknown = -1
  New = 0

  Preprocessed = 5
  InWork = 10

  Excluded = 12
  Error = 11

  Done = 15


ORG_LEVELS_names: [str] = [
  'общее собрание участников',
  'единственный участник',
  'общее собрание акционеров',
  'единственный акционер',
  'совет директоров',
  'генеральный директор',
  'правление общества']


@unique
class Currencies(Enum):
  RUB = 0
  USD = 1
  EURO = 2
  KZT = 3
  BYN = 4
  TJR = 5
  RSD = 6
  KGS = 7
  UAH = 8
  Percent = 100


currencly_map = {
  'руб': Currencies.RUB.name,
  'дол': Currencies.USD.name,
  'евр': Currencies.EURO.name,
  'тэн': Currencies.KZT.name,
  'тен': Currencies.KZT.name,

  'сомони': Currencies.TJR.name,
  'Таджикский сомони': Currencies.TJR.name,

  'гривна': Currencies.UAH.name,
  'Украинская гривна': Currencies.UAH.name,
  'гривен': Currencies.UAH.name,

  'динар': Currencies.RSD.name,
  'Сербский динар': Currencies.RSD.name,

}


@unique
class OrgStructuralLevel(Enum, metaclass=DisplayStringEnumMeta):
  # TODO: define per org_types

  AllMembers = 4, ['Общее собрание участников', 'Единственный участник']
  ShareholdersGeneralMeeting = 3, ['Общее собрание акционеров', 'Единственный акционер']
  BoardOfDirectors = 2, ['Совет директоров']
  CEO = 1, ['Генеральный директор']
  BoardOfCompany = 0, ['Правление общества']

  @staticmethod
  def get_all_display_names(nm: str) -> [str]:
    ret = []
    for x in OrgStructuralLevel:
      if isinstance(x.display_string, list):
        for ds in x.display_string:
          ret.append(ds)
      else:
        ret.append(x.display_string)

    return ret

  @staticmethod
  def find_by_display_string(nm: str) -> str or None:
    for x in OrgStructuralLevel:
      if isinstance(x.display_string, list):
        for ds in x.display_string:
          if ds.lower() == nm.lower():
            return x.name
      elif x.display_string.lower() == nm.lower():
        return x.name
    return None

  @staticmethod
  def as_db_json():
    return [{"_id": x.name, "number": x.value, "alias": x.display_string} for x in OrgStructuralLevel]


@unique
class ContractTags(Enum, metaclass=DisplayStringEnumMeta):
  Value = 0, 'value'
  Currency = 1, 'currency'
  Sign = 2, 'sign'


# @unique
# class ContractSubject(Enum, metaclass=DisplayStringEnumMeta):
#   '''
#   TODO: rename ContractSubject->DocumentSubject, because contract subjects are only a subset of this
#   '''
#   Other = -1, 'Другое'
#
#   Deal = 0, 'Сделка'
#   Charity = 1, 'Благотворительность'
#   RealEstate = 4, 'Сделки с недвижимым имуществом'
#   Loans = 7, 'Займы, кредиты и др. обязательста'


@unique
class InsiderInfoType(Enum):
  Bankruptcy = 0
  SharesOfParticipation = 1
  SEB = 2
  Changes = 3
  Licenses = 4
  OSA = 5
  Accounting = 6
  Plans = 7
  SD = 8
  Deals = 9
  Courts = 10
  IssuedSecurities = 11


@unique
class ContractSubject(Enum, metaclass=DisplayStringEnumMeta):
  Other = 0, 'Другое'

  Charity = 1, 'Благотворительность'
  RealEstate = 4, 'Сделки с недвижимым имуществом'
  Loans = 7, 'Займы, кредиты и др. обязательста'

  # Other = 2, 'Другое'
  Lawsuit = 3, 'Судебные издержки'

  Insurance = 5, 'Страхование'
  Consulting = 6, 'Консультационные услуги'
  RentingOut = 8, 'Передача в аренду'
  Renting = 9, 'Получение в аренду недвижимого имущества'
  BigDeal = 10, ' Крупная сделка'
  Deal = 11, 'Сделка'
  # 12
  # 13
  # 14
  # 15
  # 16
  # 17
  AgencyContract = 21, 'Агентский договор'
  BankGuarantees = 22, ''
  RelatedTransactions = 23, ''
  GeneralContract = 24, ''
  EmployeeContracts = 25, ''
  PledgeEncumbrance = 26, 'Залог, обременение'
  Liquidation = 27, ''
  Service = 28, ''
  CashPayments = 29, ''
  RefusalToLeaseLand = 30, ''

  DealGeneralBusiness = 31, ''
  RevisionCommission = 32, ''
  Reorganization = 33, ''
  InterestedPartyTransaction = 34, ''
  RelatedPartyTransaction = 35, ''
  AssetTransactions = 36, ''
  DealIntellectualProperty = 37, ''
  RealEstateTransactions = 38, ''
  SecuritiesTransactions = 39, ''
  RegisteredCapital = 40, ''

  ParticipationInOtherOrganizations = 41, ''
  DecisionsForSubsidiary = 42, ''

  @staticmethod
  def as_matrix():
    return np.array([[s.name, s.value] for s in ContractSubject])

  @staticmethod
  def encode_1_hot():
    '''
    bit of paranoia to reserve order
    :return:
    '''
    all_subjects_map = ContractSubject.as_matrix()
    values = all_subjects_map[:, 1]

    # encoding integer subject codes in one-hot vectors
    _cats = to_categorical(values)

    subject_name_1hot_map = {all_subjects_map[i][0]: _cats[i] for i, k in enumerate(all_subjects_map)}

    return subject_name_1hot_map


contract_subjects = [
  ContractSubject.Charity,
  ContractSubject.RealEstate,
  ContractSubject.Renting,
  ContractSubject.Deal,
  ContractSubject.Service,
  ContractSubject.Loans,
  ContractSubject.PledgeEncumbrance]
