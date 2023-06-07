from typing import Dict, List

import mlflow
import pandas as pd

from analyser.dictionaries import integration_path, labels, label2id
from utilits.utils import _env_var

model = None


def wrapper(document):
    if not document:
        raise ValueError('Empty document')

    global model

    json_from_text = concat_paragraphs_to_string(document)

    if model is None:
        TRACKING_URI = _env_var('ML_FLOW_TRACKING_URI')
        mlflow.set_tracking_uri(TRACKING_URI)

        model_name = _env_var('ML_FLOW_MODEL_NAME', "practice-classifier-ruRoberta-large")
        stage = _env_var('ML_FLOW_STAGE', "Staging")

        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}",
                                         dst_path='integration/classifier/model')

    df = pd.DataFrame([json_from_text], columns=['text'])
    predictions = model.predict(df)
    result = []
    for index, row in predictions.iterrows():
        result.append({
            'label': row['practice'],
            'score': row['confidence']
        })
    return sorted(result, key=lambda x: x['score'], reverse=True)


def concat_paragraphs_to_string(document: Dict[str, List]) -> str:
    text: str = ''
    for par in document['paragraphs']:
        text += ' ' + par['paragraphHeader']['text']
        text += ' ' + par['paragraphBody']['text']
    return text.strip()


if __name__ == '__main__':
    # TODO: remove it
    print(integration_path)
    print(labels)
    print(label2id)