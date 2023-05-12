import os
import re

from analyser.hyperparams import HyperParameters
from analyser.text_tools import compare_masked_strings

data = {
  "Subsidiary": [
    {
      "_id": "Газпром нефть",
      "legal_entity_type": "ПАО",
      "aliases": [
        "Газпром нефть"
      ]
    },
    {
      "_id": "Газпромнефть Шиппинг",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть Шиппинг"
      ]
    },
    {
      "_id": "Арктика Медиа",
      "legal_entity_type": "АО",
      "aliases": [
        "Арктика Медиа"
      ]
    },
    {
      "_id": "Издательско-полиграфическое предприятие «Мастерская печати»",
      "legal_entity_type": "ООО",
      "aliases": [
        "ИПП Мастерская печати",
        "ИПП «Мастерская печати»"
      ]
    },
    {
      "_id": "Рекламно-Информационное агентство «Город»",
      "legal_entity_type": "ООО",
      "aliases": [
        "РИА Город",
        "РИА «Город»"
      ]
    },
    {
      "_id": "Газпромнефть-Аэро",
      "legal_entity_type": "АО",
      "aliases": [
        "Газпромнефть-Аэро"
      ]
    },
    {
      "_id": "Газпромнефть Марин Бункер",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть Марин Бункер"
      ]
    },
    {
      "_id": "Газпромнефть-Битум Казахстан",
      "legal_entity_type": "ТОО",
      "aliases": [
        "Газпромнефть-Битум Казахстан"
      ]
    },
    {
      "_id": "Совхимтех",
      "legal_entity_type": "АО",
      "aliases": [
        "Совхимтех"
      ]
    },
    {
      "_id": "Газпромнефть-СМ",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-СМ",
        "Газпромнефть-смазочные материалы",
        "ГПН-СМ"
      ]
    },
    {
      "_id": "Газпромнефть Лубрикантс Италия",
      "legal_entity_type": "",
      "aliases": [
        "Газпромнефть Лубрикантс Италия"
      ]
    },
    {
      "_id": "Газпромнефть Лубрикантс Украина",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть Лубрикантс Украина"
      ]
    },
    {
      "_id": "Газпромнефть-Каталитические системы",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Каталитические системы"
      ]
    },
    {
      "_id": "Газпромнефть-Энергосервис",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Энергосервис"
      ]
    },
    {
      "_id": "Газпромнефть-Логистика",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Логистика"
      ]
    },
    {
      "_id": "Газпромнефть-Битумные материалы",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Битумные материалы"
      ]
    },
    {
      "_id": "Газпромнефть-Рязанский завод битумных материалов",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Рязанский завод битумных материалов"
      ]
    },
    {
      "_id": "Газпромнефть-Тоталь ПМБ",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Тоталь ПМБ"
      ]
    },
    {
      "_id": "НОВА-БРИТ",
      "legal_entity_type": "ООО",
      "aliases": [
        "НОВА-БРИТ"
      ]
    },
    {
      "_id": "Транс-Реал",
      "legal_entity_type": "ООО",
      "aliases": [
        "Транс-Реал"
      ]
    },
    {
      "_id": "Газпромнефть-МНПЗ",
      "legal_entity_type": "АО",
      "aliases": [
        "Газпромнефть–Московский НПЗ"
      ]
    },
    {
      "_id": "Газпромнефть-ОНПЗ",
      "legal_entity_type": "АО",
      "aliases": [
        "Газпромнефть-Омский НПЗ",
        "Газпромнефть - ОНПЗ",
        "Газпромнефть - Омский НПЗ",
      ]
    },

    {
      "_id": "Нефтехимремонт",
      "legal_entity_type": "ООО",
      "aliases": [
        "Нефтехимремонт"
      ]
    },
    {
      "_id": "РМЗ ГПН-ОНПЗ",
      "legal_entity_type": "ООО",
      "aliases": [
        "РМЗ ГПН-ОНПЗ"
        'РМЗ «ГПН-ОНПЗ»',
        "Ремонтно-механический завод «ГПН-ОНПЗ»",
        "Ремонтно-механический завод «Газпромнефть-ОНПЗ»"
      ]
    },
    {
      "_id": "Альянс-Ойл-Азия",
      "legal_entity_type": "ООО",
      "aliases": [
        "Альянс-Ойл-Азия"
      ]
    },
    {
      "_id": "Моснефтепродукт",
      "legal_entity_type": "ООО",
      "aliases": [
        "Моснефтепродукт"
      ]
    },
    {
      "_id": "Битумные Терминалы",
      "legal_entity_type": "ООО",
      "aliases": [
        "Битумные Терминалы"
      ]
    },
    {
      "_id": "БСВ-ХИМ",
      "legal_entity_type": "ООО",
      "aliases": [
        "БСВ-ХИМ"
      ]
    },
    {
      "_id": "Полиэфир",
      "legal_entity_type": "ООО",
      "aliases": [
        "Полиэфир"
      ]
    },
    {
      "_id": "МЗСМ",
      "legal_entity_type": "АО",
      "aliases": [
        "МЗСМ",
        "Газпромнефть Московский Завод Смазочных Материалов",
        "Газпромнефть МЗСМ",
        "Московский Завод Смазочных Материалов"
      ]
    },
    {
      "_id": "ИТСК",
      "legal_entity_type": "ООО",
      "aliases": [
        "ИТСК"
      ]
    },
    {
      "_id": "Ноябрьскнефтегазсвязь",
      "legal_entity_type": "ООО",
      "aliases": [
        "Ноябрьскнефтегазсвязь"
      ]
    },
    {
      "_id": "Комплекс Галерная 5",
      "legal_entity_type": "ООО",
      "aliases": [
        "Комплекс Галерная 5"
      ]
    },
    {
      "_id": "Юнифэл",
      "legal_entity_type": "ООО",
      "aliases": [
        "Юнифэл"
      ]
    },
    {
      "_id": "Многофункциональный комплекс «Лахта центр»",
      "legal_entity_type": "АО",
      "aliases": [
        "МФК Лахта Центр",
        'МФК «Лахта центр»',
        "Многофункциональный комплекс Лахта центр"
      ]
    },
    {
      "_id": "ГПН-Инвест",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Инвест"
      ]
    },
    {
      "_id": "ГПН-ЗС",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-ЗС"
      ]
    },
    {
      "_id": "Алтайское Подворье",
      "legal_entity_type": "ООО",
      "aliases": [
        "Алтайское Подворье"
      ]
    },
    {
      "_id": "ГПН-Финанс",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Финанс"
      ]
    },
    {
      "_id": "ГПН-Энерго",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Энерго"
      ]
    },
    {
      "_id": "Газпромнефть-Трейд Оренбург",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Трейд Оренбург"
      ]
    },
    {
      "_id": "ГПН-проект",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-проект"
      ]
    },
    {
      "_id": "Клуб Заречье",
      "legal_entity_type": "ООО",
      "aliases": [
        "Клуб Заречье"
      ]
    },
    {
      "_id": "Газпромнефть-Оренбург Союз",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Оренбург Союз"
      ]
    },
    {
      "_id": "Южуралнефтегаз",
      "legal_entity_type": "АО",
      "aliases": [
        "Южуралнефтегаз"
      ]
    },
    {
      "_id": "Газпромнефть-ННГ",
      "legal_entity_type": "АО",
      "aliases": [
        "Газпромнефть-ННГ",
        "Газпромнефть-Ноябрьскнефтегаз"
      ]
    },
    {
      "_id": "Газпромнефть-ННГГФ",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-ННГГФ",
        "Газпромнефть-Ноябрьскнефтегазгеофизика"
      ]
    },
    {
      "_id": "Газпромнефть Экспертные решения",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть Экспертные решения",
        "Газпромнефть-Экспертные решения"
      ]
    },
    {
      "_id": "Газпромнефть-Оренбург",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Оренбург"
      ]
    },
    {
      "_id": "Газпромнефть-Заполярье",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Заполярье"
      ]
    },
    {
      "_id": "Газпромнефть-Нефтесервис",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Нефтесервис"
      ]
    },
    {
      "_id": "Газпромнефть НТЦ",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть НТЦ",
        "Газпромнефть Научно-Технический Центр"
      ]
    },
    {
      "_id": "НоябрьскНефтеГазАвтоматика",
      "legal_entity_type": "ООО",
      "aliases": [
        "НоябрьскНефтеГазАвтоматика"
      ]
    },
    {
      "_id": "Гарант Сервис",
      "legal_entity_type": "ООО",
      "aliases": [
        "Гарант Сервис"
      ]
    },
    {
      "_id": "Газпромнефть-Ангара",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Ангара"
      ]
    },
    {
      "_id": "Газпромнефть-ГЕО",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-ГЕО"
      ]
    },
    {
      "_id": "Карабашские 6 (бывшее ООО Торг-72)",
      "legal_entity_type": "ООО",
      "aliases": [
        "Карабашские 6 (бывшее ООО Торг-72)"
      ]
    },
    {
      "_id": "Энерком",
      "legal_entity_type": "ООО",
      "aliases": [
        "Энерком"
      ]
    },
    {
      "_id": "Газпромнефть-Аэро Брянск",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Аэро Брянск"
      ]
    },
    {
      "_id": "Газпромнефть-Восток",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Восток"
      ]
    },
    {
      "_id": "ГПН-Развитие",
      "legal_entity_type": "ООО",
      "aliases": [
        "ГПН-Развитие",
        "Газпромнефть-Развитие"
      ]
    },
    {
      "_id": "Газпромнефть-Хантос",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Хантос"
      ]
    },

    {
      "_id": "Южно-Приобский ГПЗ",
      "legal_entity_type": "ООО",
      "aliases": [
        "Южно-Приобский ГПЗ",
        "Южно-Приобский газоперерабатывающий завод"
      ]
    },
    {
      "_id": "Технологический центр «Бажен»",
      "legal_entity_type": "ООО",
      "aliases": [
        "Технологический центр Бажен",
        "Технологический центр «Бажен»"
      ]
    },
    {
      "_id": "Меретояханефтегаз",
      "legal_entity_type": "ООО",
      "aliases": [
        "Меретояханефтегаз"
      ]
    },
    {
      "_id": "Нови Сад",
      "legal_entity_type": "НИС а.о.",
      "aliases": [
        "Нови Сад"
      ]
    },
    {
      "_id": "Газпромнефть-Ямал",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Ямал"
      ]
    },
    {
      "_id": "Ноябрьсктеплонефть",
      "legal_entity_type": "ООО",
      "aliases": [
        "Ноябрьсктеплонефть"
      ]
    },
    {
      "_id": "Ноябрьскэнергонефть",
      "legal_entity_type": "ООО",
      "aliases": [
        "Ноябрьскэнергонефть"
      ]
    },
    {
      "_id": "Морнефтегазпроект",
      "legal_entity_type": "АО",
      "aliases": [
        "Морнефтегазпроект"
      ]
    },
    {
      "_id": "Газпром нефть шельф",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпром нефть шельф"
      ]
    },
    {
      "_id": "Газпромнефть-Приразломное",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Приразломное"
      ]
    },
    {
      "_id": "Газпромнефть-Сахалин",
      "legal_entity_type": "ООО",
      "aliases": [
        "ГПН-Сахалин",
        "Газпромнефть-Сахалин",
        "Газпромнефть-Сахалин"
      ]
    },

    {
      "_id": "Газпромнефть Бизнес-сервис",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть Бизнес-сервис"
      ]
    },
    {
      "_id": "Газпромнефть-Центр",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Центр"
      ]
    },
    {
      "_id": "Газпромнефть-Альтернативное топливо",
      "legal_entity_type": "АО",
      "aliases": [
        "Газпромнефть-Альтернативное топливо"
      ]
    },
    {
      "_id": "Газпромнефть-Лаборатория",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Лаборатория"
      ]
    },
    {
      "_id": "Газпромнефть-Транспорт",
      "legal_entity_type": "АО",
      "aliases": [
        "Газпромнефть-Транспорт"
      ]
    },
    {
      "_id": "Газпромнефть-Терминал",
      "legal_entity_type": "АО",
      "aliases": [
        "Газпромнефть-Терминал"
      ]
    },
    {
      "_id": "Газпромнефть-Региональные продажи",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Региональные продажи"
      ]
    },
    {
      "_id": "Газпромнефть-Корпоративные продажи",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Корпоративные продажи",
        "Газпром Нефть Корпоративные продажи"
      ]
    },
    {
      "_id": "Газпромнефть-Белнефтепродукт",
      "legal_entity_type": "ИООО",
      "aliases": [
        "Газпромнефть-Белнефтепродукт"
      ]
    },
    {
      "_id": "Газпромнефть-Казахстан",
      "legal_entity_type": "ТОО",
      "aliases": [
        "Газпромнефть-Казахстан"
      ]
    },
    {
      "_id": "Газпромнефть-Таджикистан",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Таджикистан"
      ]
    },
    {
      "_id": "Газпромнефть-Новосибирск",
      "legal_entity_type": "АО",
      "aliases": [
        "Газпромнефть-Новосибирск",
        "Газпромнефть-Новосибирск (НБ)"
      ]
    },
    {
      "_id": "Газпромнефть-Урал",
      "legal_entity_type": "АО",
      "aliases": [
        "Газпромнефть-Урал"
      ]
    },
    {
      "_id": "Газпромнефть-Ярославль",
      "legal_entity_type": "АО",
      "aliases": [
        "Газпромнефть-Ярославль"
      ]
    },
    {
      "_id": "Газпромнефть-Северо-Запад",
      "legal_entity_type": "АО",
      "aliases": [
        "Газпромнефть-Северо-Запад"
      ]
    },
    {
      "_id": "Газпромнефть-Красноярск",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Красноярск"
      ]
    },
    {
      "_id": "Универсал нефть",
      "legal_entity_type": "АО",
      "aliases": [
        "Универсал нефть"
      ]
    },
    {
      "_id": "Мунай-Мырза",
      "legal_entity_type": "ЗАО",
      "aliases": [
        "Мунай-Мырза"
      ]
    },
    {
      "_id": "Газпромнефть-Мобильная карта",
      "legal_entity_type": "АО",
      "aliases": [
        "Газпромнефть-Мобильная карта"
      ]
    },
    {
      "_id": "Газпромнефть-Снабжение",
      "legal_entity_type": "ООО",
      "aliases": [
        "Газпромнефть-Снабжение"
      ]
    }
  ]
}

subsidiaries = []


def all_do_names():
  for s in subsidiaries:
    for alias in s['aliases'] + [s['_id']]:
      yield alias


def estimate_subsidiary_name_match_min_jaro_similarity():
  top_similarity = 0

  all_do_names_ = list(all_do_names())
  all_do_names_fixed = []
  for name1 in all_do_names_:
    all_do_names_fixed.append(name1.replace('»', '').replace('«', '').lower())

  for name1 in all_do_names_fixed:
    for name2 in all_do_names_fixed:

      if name1 != name2:

        similarity = compare_masked_strings(name1, name2, [])
        if similarity > top_similarity:
          top_similarity = similarity

  return top_similarity


def is_gpn_name(name):
  return re.search('Газпром', name, re.IGNORECASE)


def update_subsidiaries_cache(new_subsidiaries):
  global subsidiaries
  subsidiaries = new_subsidiaries
  HyperParameters.subsidiary_name_match_min_jaro_similarity = estimate_subsidiary_name_match_min_jaro_similarity()
  print('HyperParameters.subsidiary_name_match_min_jaro_similarity',
        HyperParameters.subsidiary_name_match_min_jaro_similarity)


if os.environ.get("GPN_CSGK_WSDL") is None:
  update_subsidiaries_cache(data['Subsidiary'])
