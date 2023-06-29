import re

ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')

list_of_address_regexes = [
  # (опционально: ООО, "Фирма"), Индекс, город, улица, дом
  # re.compile(r'((([А-Я]{2,3}|)\s+«[а-яА-Я\s\-]+»(\s+|)|)'
  #            r'\d+, (г\.|город)\s+[а-яА-Я\-]+,'
  #            r'(ул\.|\s|улица|)(\s|)[а-яА-Я\-\s]+(|\s)(проспект|улица|наб.|пр-кт|ул.|\s|),'
  #            r'(\s|)(д\.|дом)(\s|)[\d\-\/]*)', re.RegexFlag.MULTILINE | re.RegexFlag.IGNORECASE),
  re.compile(r'(([А-Я]{2,3}\s*«[\w\s\-]+?»)|(НКО[\w\s\-]*?\([А-Я]{2,3}\)))'
             r'\s*\d+.,\s*([\w\-\s]+?,\s*)*'
             r'((г\.|город|гор\.|посёлок|район|р\-н)\s*[\w\s\-]+,\s*)'
             r'([\w\s\-]*?\s*(ул\.|ул|улица|шоссе|проспект|пр\-кт|пер\.|переулок|наб\.|набережная|тракт|мкр\-н)[\w\s\-]*(\(.*?\))?)'
             r'((,\s*)(\d+|(д\.|дом|помещение|каб\.|кабинет|пом\.|помещ\.|литера|лит\.|литер|пом\.|этаж|эт|зд\.|здание|офис)\s*[\w\-\/]+)(.+?(ком|пом|помещение|литер|литера|литер\.)\s+[\w\-\/]+)?)*'
             ),

  re.compile(
    r'((Почтовый|почтовый|Юридический)\s+)?адрес\:\s*'
    r'([\w\-\s]+,\s*)*?\d+.,\s*([\w\-\s]+,\s*)*'
    r'((г\.|город|гор\.|посёлок|район|р\-н)\s*[\w\s\-]+,\s*)'
    r'([\w\s\-]*?\s*(ул\.|ул|улица|шоссе|проспект|пр\-кт|пер\.|переулок|наб\.|набережная|тракт|мкр\-н)[\w\s\-]*(\(.*?\))?)'
    r'((,\s*)(\d+|(д\.|дом|помещение|каб\.|кабинет|пом\.|помещ\.|литера|лит\.|литер|пом\.|этаж|эт|зд\.|здание|офис)\s*[\w\-\/]+)(.+?(ком|пом|помещение|литер|литера|литер\.)\s+[\w\-\/]+)?)*'
    # re.MULTILINE
  ),

  re.compile(
    r'место\s+нахождения\:\s*'
    r'([\w\-\s]+,\s*)*?\d+.,\s*([\w\-\s]+,\s*)*'
    r'((г\.|город|гор\.|посёлок|район|р\-н)\s*[\w\s\-]+,\s*)'
    r'([\w\s\-]*?\s*(ул\.|ул|улица|шоссе|проспект|пр\-кт|пер\.|переулок|наб\.|набережная|тракт|мкр\-н)[\w\s\-]*(\(.*?\))?)'
    r'((,\s*)(\d+|(д\.|дом|помещение|каб\.|кабинет|пом\.|помещ\.|литера|лит\.|литер|пом\.|этаж|эт|зд\.|здание|офис)\s*[\w\-\/]+)(.+?(ком|пом|помещение|литер|литера|литер\.)\s+[\w\-\/]+)?)*'
    # re.MULTILINE
  ),

  # Индекс, (опционально: страна), область, город, улица, дом
  re.compile(r'(\d+,((\s|)[а-яА-Я\-\s]+(\s|),){1,2}'
             r'(\s|)(г\.|город) [а-яА-Я\-]+(,|\s|)\s'
             r'(ул\.|\s|улица|)(\s|)[а-яА-Я\-\s]+(|\s)(проспект|улица|наб.|пр-кт|ул.|\s|),'
             r'(\s|)(д\.|дом)(\s|)[\d\-]+)'),

  # Улица, дом, город, индекс
  re.compile(r'((ул\.|\s|улица|)(\s|)[а-яА-Я\-\s]+(|\s)(проспект|улица|наб.|пр-кт|ул.|\s|),'
             r'(\s|)(д\.|дом)(\s|)[\d\-\/]*,'
             r'(\s|)(г\.|город)\s+[а-яА-Я\-]+,'
             r'(\s|)\d+)'),

  #     # ООО, "Фирма", Индекс, 4 блока разделенные ",", дом(помещение или что-то еще =D)
  #     re.compile(r'(([А-Я]{2,3}|)(\s+|)«[а-яА-Я\-\s]+»(\s+|)'
  #                r'\d+,(\s+[а-яА-Я\s\.\-\d\(\)]+,){4,}[а-яА-Я\s\.\d]+[\d\sа-яА-Я]+)'),

  #     # Индекс, 4 блока разделенные ",", дом(помещение или что-то еще =D)
  #     re.compile(r'(\d+,(\s+[а-яА-Я\s\.\-\d\(\)]+,){4,}[а-яА-Я\s\.\d]+[\d\sа-яА-Я]+)'),
]

list_of_number_regex = [
  re.compile(r'(\s+(ИНН|ОГРН|ОКПО|КПП|УИН|БИК|ОКТМО)\s+\d+(\s+|,|\.))'),
  # (ИНН: 9999999999999)
  re.compile(r'(\((ИНН|ОГРН|ОКПО|КПП|УИН|БИК|ОКТМО):\s*\d+(\s+|)\))'),
  # (ОГРН: 9999999999999
  re.compile(r'(\((ИНН|ОГРН|ОКПО|КПП|УИН|БИК|ОКТМО):\s*\d+)'),
  # ОГРН: 06.06.2005
  re.compile(r'((ИНН|ОГРН|ОКПО|КПП|УИН|БИК|ОКТМО)(:|)\s*[\d\.]+)'),
  re.compile(r'((Р\/с|К\/с|л\/с|Р\/счет)\s*(\№)?\s*\d+)', re.RegexFlag.IGNORECASE)
]

list_of_phone_regexes = [
  # т. (9999) 99-99-99
  re.compile(r'(\sт.\s+\(\d{3,4}\)\s\d\d(\-|\s)\d\d(\-|\s)\d\d(\,|\s))'),
  # т./факс 99-99-99
  re.compile(r'(\s+(т|тел)\s*(\.|)\/факс(\:|)\s* (\(|)\d+(\)|)((\-|\s)\d+){3})', re.RegexFlag.IGNORECASE),
  # Телефон: +7 (9999) 99-99-99
  re.compile(r'(\s*(Телефон|Тел\.):\s*(\+|)\d\s*\(\d+\)([\s\-]\d+){3})', re.RegexFlag.IGNORECASE),
  # 8 (999) 999-99-99
  re.compile(r'((\+|)\d\s+\(\d{3}\)\s+\d{3}(\-|\s)\d{2}(\-|\s)\d{2})'),
  # тел.: 8-999-99-999-99
  re.compile(r'(\s*(Телефон|тел\.):\s*(\+|)\d\-\s*\d{3}([\s\-]\d+){3})', re.RegexFlag.IGNORECASE),
]


def clear_text(text: str) -> str:
  text = re.sub(r'(?<=[^\.\?\;\:\!])(?<=[^ \t\r\f\v])\n{2}', '.\n', text)
  text = re.sub(r'\n{3,}', '\n\n', text)
  text = re.sub(r'[ \t\r\f\v]{2,}', ' ', text)
  text = re.sub(r'(([а-яА-Яa-zA-Z\d\s\u0000-\u26FF]{1,2}( |\s)){5,})', ' ', text)

  text = ILLEGAL_CHARACTERS_RE.sub('', text)

  bad_symbols = ['_x000D_', '\x07', r'FORM[A-Z]+',
                 '\u0013', '\u0001', '\u0014', '\u0015', '\u0007', '<', '>', '_+', '']
  for bad_symbol in bad_symbols:
    text = re.sub(bad_symbol, ' ', text)

  return text


def remove_header(text: str) -> str:
  for number in list_of_number_regex:
    text = number.sub(' ', text)

  for address in list_of_address_regexes:
    text = address.sub(' ', text)

  for phone in list_of_phone_regexes:
    text = phone.sub(' ', text)

  return text


def cleanup_all(t):
  t = clear_text(t)
  t = remove_header(t)
  return t