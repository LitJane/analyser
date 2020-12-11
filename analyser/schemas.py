#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


# schemas.py
import warnings
from enum import Enum

from analyser.ml_tools import SemanticTagBase
from analyser.structures import OrgStructuralLevel, ContractSubject, currencly_map

tag_value_field_name = "value"


class DocumentSchema:
  date: SemanticTagBase or None = None
  number: SemanticTagBase or None = None

  def __init__(self):
    super().__init__()


class HasOrgs:
  def __init__(self):
    super().__init__()
    self.orgs: [OrgItem] = []


class ContractPrice(SemanticTagBase):
  def __init__(self):
    super().__init__()

    self.amount: SemanticTagBase  # netto or brutto #deprecated
    self.currency: SemanticTagBase
    self.sign: SemanticTagBase
    self.vat: SemanticTagBase  # number
    self.vat_unit: SemanticTagBase  # percentage
    self.value_brutto: SemanticTagBase  # netto + VAT
    self.value_netto: SemanticTagBase  # value before VAT


class AgendaItemContract(HasOrgs, SemanticTagBase):
  number: SemanticTagBase = None
  date: SemanticTagBase = None
  price: ContractPrice = None

  def __init__(self):
    super().__init__()


class AgendaItem(SemanticTagBase):


  def __init__(self, tag= None):
    super().__init__(tag)
    self.solution: SemanticTagBase or None = None

    self.contract: AgendaItemContract = AgendaItemContract()


class OrgItem():

  def __init__(self):
    super().__init__()
    self.type: SemanticTagBase or None = None
    self.name: SemanticTagBase or None = None
    self.alias: SemanticTagBase or None = None  # a.k.a role in the contract
    self.alt_name: SemanticTagBase or None = None  # a.k.a role in the contract

  def as_list(self) -> [SemanticTagBase]:
    warnings.warn("use OrgItem", DeprecationWarning)
    return [getattr(self, key) for key in ["type", "name", "alias"] if getattr(self, key) is not None]


class ContractSchema(DocumentSchema, HasOrgs):
  price: ContractPrice = None

  def __init__(self):
    super().__init__()
    self.subject: SemanticTagBase or None = None


class ProtocolSchema(DocumentSchema):

  def __init__(self):
    super().__init__()
    self.org: OrgItem = OrgItem()
    self.structural_level: SemanticTagBase or None = None
    self.agenda_items: [AgendaItem] = []


# class CharterConstraint:
#   def __init__(self):
#     super().__init__()
#     self.margins: [ContractPrice] = []


class Competence(SemanticTagBase):
  """
  child of CharterStructuralLevel
  """

  def __init__(self, tag: SemanticTagBase = None):
    super().__init__(tag)
    self.constraints: [ContractPrice] = []


class CharterStructuralLevel(SemanticTagBase):
  def __init__(self, tag: SemanticTagBase = None):
    super().__init__(tag)
    self.competences: [Competence] = []


class CharterSchema(DocumentSchema):
  org: OrgItem = OrgItem()

  def __init__(self):
    super().__init__()
    self.structural_levels: [CharterStructuralLevel] = []


document_schemas = {
  "$schema": "http://json-schema.org/draft-04/schema#",
  "title": "Legal document attributes",
  "description": "Legal document attributes. Schema draft 4 is used for compatibility with Mongo DB",

  "definitions": {

    "tag": {
      "description": "a piece of text, denoting an attributes",
      "type": "object",

      "properties": {
        "span": {
          "type": "array",
          "minItems": 2,
          "maxItems": 2
        },
        "confidence": {
          "type": "number"
        },
        "span_map": {
          "type": "string"
        }
      },
      "required": ["span", tag_value_field_name]
    },

    "string_tag": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            tag_value_field_name: {
              "type": "string"
            }
          },

        }]
    },

    "boolean_tag": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            tag_value_field_name: {
              "type": "boolean"
            }
          },

        }]

    },

    "number_tag": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            tag_value_field_name: {
              "type": "number"
            }
          }
        }],
    },

    "date_tag": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            tag_value_field_name: {
              "type": "string",
              "format": "date-time"
            }
          },

        }],

    },

    "agenda_contract": {
      "description": "Атрибуты контракта, о котором идет речь в повестке",

      "properties": {

        "number": {
          "$ref": "#/definitions/string_tag"
        },

        "date": {
          "$ref": "#/definitions/date_tag"
        },

        "solution": {
          "$ref": "#/definitions/boolean_tag"
        },

        "warnings": {
          "type": "string"
        },

        "orgs": {
          "type": "array",
          "items": {
            "$ref": "#/definitions/contract_agent",
          }
        },

        "value": {
          "$ref": "#/definitions/currency_value"
        },

      },
      "additionalProperties": False

    },
    "agenda": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            "contract": {"$ref": "#/definitions/agenda_contract"}
          },
          "required": ["span"],
          # "additionalProperties": False
        }],
    },

    "currency": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            tag_value_field_name: {
              "enum": list(currencly_map.values())
            }
          }
        }],
    },

    "sign": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            tag_value_field_name: {
              "type": "integer",
              "enum": [-1, 0, 1]
            }
          }}]
    },

    "currency_value": {

      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            "value": {
              "$ref": "#/definitions/number_tag",
            },

            "currency": {
              "$ref": "#/definitions/currency"
            },

            "sign": {
              "$ref": "#/definitions/sign",
            }

          },
          "required": ["sign", "value", "currency"],
        }
      ],

    },

    "competence": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {
          "properties": {
            tag_value_field_name: {
              "enum": ContractSubject._member_names_
            },
            "constraints": {
              "type": "array",
              "items": {"$ref": "#/definitions/currency_value"}
            }
          }}],
    },

    "structural_level": {
      "allOf": [
        {"$ref": "#/definitions/tag"},
        {"properties": {
          tag_value_field_name: {
            "enum": OrgStructuralLevel._member_names_
          },

          "competences": {
            "type": "array",
            "items": {"$ref": "#/definitions/competence"},
          }
        }},
      ],

    },

    "org": {
      "type": "object",
      "properties": {
        "name": {"$ref": "#/definitions/string_tag"},
        "type": {"$ref": "#/definitions/string_tag"}
      },
      # "required": ["name", "type"]
    },

    "contract_agent": {
      "allOf": [
        {"$ref": "#/definitions/org"},
        {
          "properties": {
            "alias": {"$ref": "#/definitions/string_tag"},
          }
        }
      ]

      # "required": ["name", "type"]
    }
  },

  "properties": {

    "charter": {
      "properties": {

        "date": {
          "$ref": "#/definitions/date_tag"
        },

        "orgs": {
          "type": "array",
          "maxItems": 1,
          "uniqueItems": True,
          "items": {
            "$ref": "#/definitions/org",
          }
        },

        "structural_levels": {
          "type": "array",
          "uniqueItems": True,
          "items": {
            "$ref": "#/definitions/structural_level"
          }
        }
      }
    },

    "contract": {
      "properties": {

        "date": {
          "$ref": "#/definitions/date_tag"
        },

        "number": {
          "$ref": "#/definitions/number_tag"
        },

        "orgs": {
          "type": "array",
          "maxItems": 10,
          "uniqueItems": True,
          "items": {
            "$ref": "#/definitions/org",
          }
        },

      }
    },

    "protocol": {
      "properties": {

        "date": {
          "$ref": "#/definitions/date_tag"
        },

        "number": {
          "$ref": "#/definitions/number_tag"
        },

        "structural_level": {
          "$ref": "#/definitions/structural_level"
        },

        "orgs": {
          "type": "array",
          "maxItems": 1,
          "items": {
            "$ref": "#/definitions/org",
          }
        },

        "agenda_items": {
          "type": "array",
          "items": {
            "$ref": "#/definitions/agenda",
          }
        },

      },
      "additionalProperties": False
    },

  },

}


# ---------------------------
# self test
# validate(instance={"date":{}}, schema=charter_schema)


class Schema2LegacyListConverter:

  def __init__(self):

    self.attr_handlers = [
      self.handleCharterStructuralLevel,
      self.handleCompetence,
      self.handleContractPrice,
      self.handleValueTag,
      self.handleCharterOrgItem]

  @staticmethod
  def handleCharterStructuralLevel(tag, key, parent_key):
    if isinstance(tag, CharterStructuralLevel):
      return tag.value.name

  @staticmethod
  def handleCompetence(tag, key, parent_key):
    if isinstance(tag, Competence):
      return tag.value.name

  @staticmethod
  def handleValueTag(tag, key, parent_key):
    if key == 'amount':
      return 'value'

  @staticmethod
  def handleCharterOrgItem(tag, key, parent_key):
    if parent_key == 'org':
      if key in ['type', 'name', 'alias']:
        return None, f"org-1-{key}"

  @staticmethod
  def handleContractPrice(tag, key, parent_key):
    if isinstance(tag, ContractPrice):
      suffix = 'min'
      if hasattr(tag, 'sign'):
        amnt = tag.sign.value
        if amnt < 0:
          suffix = "max"

      return f"constraint-{suffix}"

  def key_of_attr(self, tag: SemanticTagBase, key, parent_key=None, index=-1) -> (str, str):
    ret = key
    for handler in self.attr_handlers:
      s = handler(tag, key, parent_key)
      if isinstance(s, tuple):
        parent_key = s[0]
        s = s[1]
      if s is not None:
        ret = s
        break

    if index != -1:
      if not isinstance(tag.value, Enum):
        # do not number enums
        ret = f'{ret}-{index + 1}'

    return parent_key, ret

  def tag_to_attr(self, tag: SemanticTagBase, key: str = "", parent_key=None, index=-1):
    v = tag.value
    if isinstance(v, Enum):
      v = v.name

    parent_key, self_key = self.key_of_attr(tag, key, parent_key, index)
    if parent_key:
      full_key = f'{parent_key}/{self_key}'
    else:
      full_key = self_key
    # full_key = self_key

    ret = {}
    ret['value'] = v
    if hasattr(tag, "confidence"):
      ret['confidence'] = tag.confidence
    if hasattr(tag, "span"):
      ret['span'] = tag.span
    if parent_key is not None:
      ret['parent'] = parent_key

    return full_key, ret

  def schema2list(self, dest: dict, d, attr_name: str = None, parent_key=None, index=-1):
    _key = attr_name

    if not hasattr(d, '__dict__'):
      return

    if isinstance(d, SemanticTagBase):
      # print("\t\t\t >>> TAG", d.value, type(d))
      _key, v = self.tag_to_attr(d, attr_name, parent_key, index)
      dest[_key] = v

    # dig into attributes
    for a_name, attr_value in vars(d).items():

      if isinstance(attr_value, list):
        # print(f"\t\t\t\n [{attr}]...")
        for i, itm in enumerate(attr_value):
          self.schema2list(dest, itm, attr_name=a_name, parent_key=_key, index=i)

      elif isinstance(attr_value, object) and not a_name.startswith('_'):
        # print("OBJET", a_name, type(attr_value), type(d))
        self.schema2list(dest, attr_value, attr_name=a_name, parent_key=_key)
      # elif isinstance(v, dict):
      #   pass
