#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from enum import Enum, unique, EnumMeta

legal_entity_types = {
  'Акционерное общество': 'АО',
  'Публичное акционерное общество': 'ПАО',
  'Общество с ограниченной ответственностью': 'ООО',
  'Иностранное общество с ограниченной ответственностью': 'ИООО',
  'Товарищество с ограниченной ответственностью': 'ТОО',
  'Товарищество с ограниченной ответственностью и какашками': 'ТОО',
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
class OrgStructuralLevel(Enum, metaclass=DisplayStringEnumMeta):
  # TODO: define per org_types

  AllMembers = 4, 'Общее собрание Участников'
  ShareholdersGeneralMeeting = 3, 'Общее собрание акционеров'
  BoardOfDirectors = 2, 'Совет директоров'
  CEO = 1, 'Генеральный директор'
  BoardOfCompany = 0, 'Правление общества'

  @staticmethod
  def find_by_display_string(nm: str) -> str or None:
    for x in OrgStructuralLevel:
      if x.display_string == nm:
        return x.name
    return None


ORG_LEVELS_names = [x.display_string for x in OrgStructuralLevel]


@unique
class ContractTags(Enum, metaclass=DisplayStringEnumMeta):
  Value = 0, 'value'
  Currency = 1, 'currency'
  Sign = 2, 'sign'


@unique
class ContractSubject(Enum, metaclass=DisplayStringEnumMeta):
  '''
  TODO: rename ContractSubject->DocumentSubject, because contract subjects are only a subset of this
  '''
  Deal = 0, 'Сделка'
  Charity = 1, 'Благотворительность'
  Other = 2, 'Другое'
  Lawsuit = 3, 'Судебные издержки'
  RealEstate = 4, 'Недвижимость'


@unique
class CharterSubject(Enum, metaclass=DisplayStringEnumMeta):
  Deal = 0, 'Сделка'
  Charity = 1, 'Благотворительность'
  Other = 2, 'Другое'
  Lawsuit = 3, 'Судебные издержки'
  RealEstate = 4, 'Недвижимость'
  Insurance = 5, 'Страхование'
  Consulting = 6, 'Консультационные услуги'