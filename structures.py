#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from enum import Enum, unique, EnumMeta
from typing import List

from ml_tools import TokensWithAttention

org_types = {
  'org_unknown': 'undefined',
  'org_ao': 'Акционерное общество',
  'org_zao': 'Закрытое акционерное общество',
  'org_oao': 'Открытое акционерное общество',
  'org_ooo': 'Общество с ограниченной ответственностью',
  'org_nc': 'Некоммерческая организация'
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
  ShareholdersGeneralMeeting = 3, 'Генеральное собрание акционеров'
  BoardOfDirectors = 2, 'Совет директоров'
  CEO = 1, 'Генеральный директор'
  BoardOfCompany = 0, 'Правление общества'


ORG_LEVELS_names = [x.display_string for x in OrgStructuralLevel]

ORG_2_ORG = {
  'all': OrgStructuralLevel.ShareholdersGeneralMeeting,
  'gen': OrgStructuralLevel.CEO,
  'directors': OrgStructuralLevel.BoardOfDirectors,
  'pravlenie': OrgStructuralLevel.BoardOfCompany,
  # TODO: what?
  'head.all': OrgStructuralLevel.ShareholdersGeneralMeeting,
  'head.gen': OrgStructuralLevel.CEO,
  'head.directors': OrgStructuralLevel.BoardOfDirectors,
  'head.pravlenie': OrgStructuralLevel.BoardOfCompany
}


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


class Citation:
  def __init__(self, cite: TokensWithAttention = None) -> None:
    self.citation = cite


class CharterConstraint(Citation):
  def __init__(self, upper, lower, subject: ContractSubject, level: OrgStructuralLevel, *args, **kwargs) -> None:
    super(CharterConstraint, self).__init__(*args, **kwargs)
    self.upper, self.lower = upper, lower
    self.subject = subject
    self.level = level

  def __repr__(self) -> str:
    return f'Ограничение: >{self.lower} <{self.upper} :{self.level.display_string}'

  def in_range(self, value):
    return self.upper <= value <= self.lower


class Charter(Citation):
  def __init__(self, org_name, constraints: List[CharterConstraint], date=None, *args, **kwargs) -> None:
    super(Charter, self).__init__(*args, **kwargs)
    self.org_name = org_name
    self.date = date
    self.constraints = constraints

  # sorted!
  def find_constraints(self, subject: ContractSubject, value=None):
    l = []
    if value is None:
      l = [c for c in self.constraints if c.subject == subject]
    else:
      l = [c for c in self.constraints if c.subject == subject and c.in_range(value)]

    return sorted(l, key=lambda c: c.subject.value, reverse=True)


class Contract(Citation):
  def __init__(self, org_name: str, subject: ContractSubject, sum: float, contractor_name: str = None,
               date=None, *args, **kwargs) -> None:
    super(Contract, self).__init__(*args, **kwargs)
    self.org_name = org_name
    self.subject = subject
    self.value = sum

    self.contractor_name = contractor_name  # FFU
    self.date = date  # FFU


class Protocol(Contract):
  def __init__(self, *args, **kwargs) -> None:
    super(Protocol, self).__init__(*args, **kwargs)


class FinalViolationLog:

  def __init__(self, charter: Charter) -> None:
    self.charter = charter

    self.need_protocol = False
    self.has_violations = False

  def check_contract(self, contract: Contract):
    self.contract = contract

    contract_constraint: List[CharterConstraint] = self.charter.find_constraints(self.contract.subject,
                                                                                 self.contract.value)

    if contract_constraint:
      if contract_constraint[0].level > OrgStructuralLevel.CEO:
        self.need_protocol = True
        print('Нужен протокол!')  # TODO debug print
      else:
        self.has_violations = False
        print(f'Нарушений нет: '
              f'Уровень:{contract_constraint[0].level.display_string} '
              f'сумма: {contract_constraint[0].lower}<{self.contract.value}<{contract_constraint[0].upper}')  # TODO debug print
    else:
      self.has_violations = True
      print(
        f'Нарушение(?): Не найдено ни одного ограничения для суммы договора в {self.contract.value}')  # TODO debug print

  def check_protocol(self, protocol: Protocol):
    self.protocol = protocol

    if self.contract.subject == self.protocol.subject:

      if self.contract.value <= self.protocol.value:
        print('Нарушений нет!')
        return True
      else:
        print('Сумма протокола меньше суммы договора!')
        return False

    else:
      print('Предмет договора и контракта не совпадают!')

  def render(self):
    pass


if __name__ == '__main__':
  l = [
    CharterConstraint(1000, 100, ContractSubject.Charity, OrgStructuralLevel.BoardOfCompany, cite=None),
    CharterConstraint(10, 1, ContractSubject.Other, OrgStructuralLevel.ShareholdersGeneralMeeting, cite=None),
    CharterConstraint(100, 10, ContractSubject.Charity, OrgStructuralLevel.CEO, cite=None),
    CharterConstraint(100, 10, ContractSubject.Charity, OrgStructuralLevel.ShareholdersGeneralMeeting, cite=None),
  ]

  c = Charter('dd', l, cite=None)
  print(c.find_constraints(ContractSubject.Charity))

  print(f'{ContractSubject.Charity.name}')
  print(f'{ContractSubject.Charity.display_string}')
  print('All:')
  for subj in ContractSubject:
    print(f'{subj.display_string}')

  for level in OrgStructuralLevel:
    print(f'{level.display_string}')
