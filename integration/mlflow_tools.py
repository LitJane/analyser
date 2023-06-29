import json

import mlflow

import gpn_config
from analyser.hyperparams import work_dir
from analyser.log import logger
from gpn_config import configured

mlflow.set_tracking_uri(configured('MLFLOW_URL'))

mlflow_model_name = configured('MLFLOW_ANALYSER_MODEL_NAME', 'Analyser')


def __load_weights_from_mlflow(weights_file_name='make_att_model_03.h5', model_name=mlflow_model_name):
  logger.debug(configured('MLFLOW_URL'))

  mlflow_client = mlflow.tracking.MlflowClient()
  logger.info('ML FLOW tracking_uri: %s', mlflow_client.tracking_uri)
  logger.info('ML FLOW version: %s', mlflow.__version__)

  try:
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
      _d = dict(mv)
      print(_d)

      if _d['current_stage'] == configured('MLFLOW_ANALYSER_MODEL_STAGE'):
        run_id = _d['run_id']
        dst_path = work_dir / 'models' / run_id
        dst_path.mkdir(parents=True, exist_ok=True)

        with open(str(dst_path / 'current_model.json'), 'w') as f:
          json.dump(_d, f)

        tmp_path = dst_path / weights_file_name

        if tmp_path.is_file():
          # cache mekanismus:
          logger.info('model is already downloaded from MLFLOW: %s; run_id: %s', tmp_path, run_id)
        else:
          logger.info('....downloading model artifacts from MLFLOW; dest: %s; run_id: %s', tmp_path, run_id)
          tmp_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{weights_file_name}",
                                                         dst_path=dst_path)

          logger.info('weights saved to %s', tmp_path)

        return tmp_path
  except Exception as err:
    logger.exception(err)
    raise ConnectionError(f'cannot communicate to mlflow at {mlflow_client.tracking_uri}', err)


def __load_weights_from_mlflow_mock():
  logger.warning(' ⚠️ ⚠️ ⚠️ NOT using STAGED model from MLFLOW in TEST mode ⚠️ ⚠️ ⚠️  MLFLOW is not available')


if gpn_config.in_test_mode:
  load_weights_from_mlflow = __load_weights_from_mlflow_mock
else:
  load_weights_from_mlflow = __load_weights_from_mlflow

if __name__ == '__main__':
  _w = load_weights_from_mlflow()
  print(_w)
