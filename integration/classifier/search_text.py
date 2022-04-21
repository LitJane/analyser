import enum
import json
import logging
import os.path
import re
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from nltk.tokenize import WhitespaceTokenizer

all_key = {
    "CONTRACT": [
        'предмет договра', 'предмет договора', 'Предмет контракта',
        'Предмет догов', 'Предмет и общие условия договора',
        'Общие ', 'Общие сведения', 'Общие положение', 'Статья'
    ],
    "AGREEMENT": [
        'Предмет соглашения', 'Общие ', 'Общие сведения', 'Общие положение', 'определил:', 'Статья'
    ],
    "SUPPLEMENTARY_AGREEMENT": [],
    "POWER_OF_ATTORNEY": ['уполномочивает', 'предоставляет', 'назначает', 'доверенность']
}

all_bad_keys = ['Термины и определения', 'Термин', 'определения', 'Содержание']

all_good_keys = ['Цели и задачи']

integration_path = 'integration/classifier'

with open(os.path.join(integration_path, 'practices.json'), encoding='utf-8') as practice_json_file:
    all_labels = json.load(practice_json_file)

labels = list(map(lambda item: item['label'], filter(lambda item: item['auto-classified'], all_labels)))

label2id = {item['label']: item['_id'] for item in all_labels}

model = None
tokenizer = None
path_to_model = os.path.join(integration_path, 'doc-classification')
model_checkpoint2 = "sberbank-ai/ruRoberta-large"
with open(os.path.join(integration_path, 'keys_from_documents.json'), encoding='utf-8') as json_file_with_key:
    key_data = json.load(json_file_with_key)


class list_of_sheets(enum.Enum):
    GOOD = 0
    BAD = 1
    TEST = 2
    TEST2 = 3


def wrapper(document):
    """
    document: Документ от парсера

    Returns: Массив из практик отсортированных по наибольшему проценту
    """
    if not document:
        return None

    global model
    global tokenizer

    json_from_text, sheet = get_text(document, path='There\\is\\nothing\\here')

    if json_from_text is None or json_from_text['text'] == '':
        return None

    if tokenizer is None and model is None:
        if os.path.exists(path_to_model) and os.path.exists(os.path.join(path_to_model, 'config.json')) and os.path.exists(os.path.join(path_to_model, 'tf_model.h5')):
            model = TFAutoModelForSequenceClassification.from_pretrained(
                str(path_to_model), num_labels=len(labels), from_pt=False
            )
            tokenizer = AutoTokenizer.from_pretrained(str(model_checkpoint2))
        else:
            logging.error('Document classification model is not found. To enable document classification put files config.json and tf_model.h5 in integration/classifier/doc-classification')
    result_from_tokenizer = tokenizer(json_from_text['text'], truncation=True, max_length=512)
    predictions = model.predict([result_from_tokenizer['input_ids']])['logits']
    predictions = tf.nn.softmax(predictions, name=None)[0].numpy()
    result = []
    for index, item in enumerate(predictions):
        result.append({
            'id': label2id[labels[index]],
            'label': labels[index],
            'score': item.item()
        })
    return sorted(result, key=lambda x: x['score'], reverse=True)


def get_text(document, filename: str = "", path: str = ""):
    text: str = ""
    for ind, par in enumerate(document['paragraphs']):
        if ind < 2:
            text += ' ' + document['paragraphs'][ind]['paragraphHeader']['text']
            text += ' ' + document['paragraphs'][ind]['paragraphBody']['text']
        text += ' ' + document['paragraphs'][ind]['paragraphBody']['text']

    text = clear_text(text)
    text = remove_signature(text)
    text, is_cut_off = remove_header(text)
    text = remove_footer(text)
    text = remove_equal(text)

    list_of_tokenize_words: [str] = WhitespaceTokenizer().tokenize(text)
    if len(list_of_tokenize_words) >= 300 and not is_cut_off:
        text = ' '.join(list_of_tokenize_words[50:450])
    else:
        text = ' '.join(list_of_tokenize_words[:450])

    validation, length, words_length = basic_text_validation(text)
    return {
               "path": path,
               "documentType": document["documentType"],
               "name": filename if not path else path.split("\\")[-1],
               "text": text,
               "length": len(text),
               "characterLength": length,
               "wordsLength": words_length,
           }, list_of_sheets.GOOD if validation else list_of_sheets.BAD


def clear_text(text: str) -> str:
    text = re.sub(r'\s', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'(([а-яА-Яa-zA-Z\d\s\u0000-\u26FF]{1,2}( |\s)){5,})', '', text)

    bad_symbols = ['_+', '_x000D_', '\x07', 'FORMTEXT', 'FORMDROPDOWN',
                   '\u0013', '\u0001', '\u0014', '\u0015', '\u0007', '<', '>']
    for bad_symbol in bad_symbols:
        text = re.sub(bad_symbol, '', text)
    return text


def remove_header(text: str) -> (str, bool):
    text = re.sub(r'(\d+, г\. [а-яА-Я\-]+, (ул\.| |)( |)[а-яА-Я\-]+(| )(проспект|улица|| ),( |)д\.( |)[\d\-]+)', ' ',
                  text)
    text = re.sub(
        r'(\d+,(| )[а-яА-Я\- ]+(| ),(| )[а-яА-Я\-]+(| ),'
        r'(| )г\. [а-яА-Я\-]+(|,| ) (ул\.| |)( |)[а-яА-Я\-]+(| |,)( |)д\.( |)[\d\-]+)',
        ' ',
        text)
    text = re.sub(r'(\s+(ИНН|ОГРН|ОКПО|КПП)\s+\d+(\s+|,|\.))', ' ', text)
    text = re.sub(r'(\((ИНН|ОГРН|ОКПО|КПП)\s+\d+\))', ' ', text)
    text = re.sub(r'((ИНН|ОГРН|ОКПО|КПП)\s+\d+)', ' ', text)
    text = re.sub(
        r'((\s+|^)[а-яА-Я\- ]+, \d+, [а-яА-Я]+\. [а-яА-Я]+, г\. [а-яА-Я\-]+, '
        r'(ул\.| |)( |)[а-яА-Я\-]+(,|\.)( |)д\.( |)[\d\-а-яА-Я]+)',
        '', text)
    text = re.sub(r'((\s+|^)[а-яА-Я\- ]+, \d+, [а-яА-Я]+,\s+[а-яА-Я\s]+,\s+г\.\s+[а-яА-Я\s]+,'
                  r'\s+(ул\.|пр\.)\s+[а-яА-Я\s\d]+,\s+(д\.|дом)\s+\d+(\s+|)[а-яА-Я])', ' ', text)

    text = re.sub(r'(\d+,(|\s+)[а-яА-Я\- ]+(|\s+),(\s+|)(г\.|город) [а-яА-Я\-]+(|,|\s+)\s(ул\.|\s+|улица)'

                  r'( |)[а-яА-Я\-]+(| |,)(\s+|)(д\.|дом)(\s+|)[\d\-]+(\s+|)[а-яА-Я\-])', ' ', text)
    text = re.sub(r'(\d+,(\s+[а-яА-Я\s\.\-\d]+,){4,}[а-яА-Я\s\.]+[\d\sа-яА-Я]+)', ' ', text)

    phrase = re.findall(r'((?i)((\s+|^)([\d\.]{2,4})\s+Предмет Договора))', text)

    if phrase:
        try:
            return ' '.join(text.split(phrase[0][0])[1:]), True
        except Exception as e:
            print(phrase)
    else:
        number = re.findall(r'(\d\.\d\.)', text)
        if number:
            return ' '.join(text.split(number[0])[1:]), True
    return text, False


def basic_text_validation(text: str) -> (bool, int, int):
    basic_text_in_low_reg = text.lower()
    length = len(basic_text_in_low_reg)
    words_length = len(WhitespaceTokenizer().tokenize(basic_text_in_low_reg))
    return length >= 200 and words_length >= 20, length, words_length


def remove_footer(text: str) -> str:
    for key in ['С уважением', 'Приложение:', 'ЗАКАЗЧИК:', 'ПОДРЯДЧИК:']:
        array_of_text = re.split(f"(?i)({key})", text)
        if array_of_text and len(array_of_text) > 1:
            text = ''.join(array_of_text[:-2])
    return text


def remove_equal(text: str) -> str:
    text = text.strip()
    if text.startswith('='):
        text = text[1:]
    return text


def remove_signature(text: str) -> str:
    for key in ["Подписи Сторон:"]:
        if key.lower() in text.lower():
            split_text = re.split(f"(?i)({key})", text)
            text = ' '.join(split_text[:-2])
    return text
