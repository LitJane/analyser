from typing import Dict, List

import mlflow
import pandas as pd

# TODO: why importing it here?
from analyser.dictionaries import integration_path, labels, label2id
from gpn_config import configured

model = None


def wrapper(document):
    if not document:
        raise ValueError('Empty document')

    global model

    json_from_text = concat_paragraphs_to_string(document)

    if model is None:
        mlflow.set_tracking_uri(configured ('MLFLOW_URL'))

        model_name = configured('MLFLOW_CLASSIFIER_MODEL_NAME', "practice-classifier-ruRoberta-large")
        stage = configured('MLFLOW_CLASSIFIER_MODEL_STAGE', "Staging")

        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}",
                                         dst_path='integration/classifier/model')

    df = pd.DataFrame([json_from_text], columns=['text'])
    predictions = model.predict(df)
    result = []

    for _, row in predictions.iterrows():

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