import json

import mlflow

import gpn_config
from analyser.hyperparams import work_dir
from analyser.log import logger

mlflow.set_tracking_uri(gpn_config.configured('MLFLOW_URL'))


def load_weights_from_mlflow(weights_file_name='make_att_model_03.h5'):
  logger.info(gpn_config.configured('MLFLOW_URL'))

  mlflow_client = mlflow.tracking.MlflowClient()
  print('ML FLOW tracking_uri', mlflow_client.tracking_uri)
  print('ML FLOW version', mlflow.__version__)
  try:
    for mv in mlflow_client.search_model_versions("name='Analyser'"):
      _d = dict(mv)
      print(_d)

      if _d['current_stage'] == 'Production':
        run_id = _d['run_id']
        print(run_id)

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


def load_weights_from_mlflow_mock():
  logger.warning(' ⚠️ ⚠️ ⚠️ We are not using BEST model from MLFLOW in TEST mode ⚠️ ⚠️ ⚠️  MLFLOW is not available')
  return None


if gpn_config.in_test_mode:
  load_weights_from_mlflow = load_weights_from_mlflow_mock

if __name__ == '__main__':
  load_weights_from_mlflow()
