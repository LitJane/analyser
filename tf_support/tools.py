import os
from pathlib import Path

import pandas as pd
import tensorflow.keras.backend as K
from pandas import DataFrame
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from analyser.hyperparams import models_path
from analyser.log import logger


class KerasTrainingContext:

  def __init__(self, checkpoints_path=models_path, session_index=0):
    self.session_index = session_index
    self.HISTORIES = {}
    self.model_checkpoint_path = Path(checkpoints_path)
    self.EVALUATE_ONLY = True
    self.EPOCHS = 18
    self.trained_models = {}
    self.validation_steps = 1
    self.steps_per_epoch = 1

    self.reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1E-6, verbose=1)
    logger.info(f"model_checkpoint_path: {checkpoints_path}")

  def set_batch_size_and_trainset_size(self, batch_size: int, test_samples: int, train_samples: int) -> None:
    self.steps_per_epoch = max(1, int(train_samples / batch_size))
    self.validation_steps = max(1, self.steps_per_epoch // 2)

    print(f'batch_size:\t{batch_size}')
    print(f'train_samples:\t{train_samples}')
    print(f'test_samples:\t{test_samples}')
    print(f'steps_per_epoch:\t{self.steps_per_epoch}')
    print(f'validation_steps:\t{self.validation_steps}')

  def get_stats_df(self) -> (DataFrame, str):
    stats_path = os.path.join(self.model_checkpoint_path, 'train_statistics0.csv')

    try:
      stats = pd.read_csv(stats_path, index_col='model_name')
    except:
      print(f'🦖🦖🦖cannot read {stats_path}')
      stats = pd.DataFrame(columns=['model_name', 'epoch', 'val_acc', 'val_loss', 'loss', 'acc']).set_index(
        'model_name')

    stats.to_csv(stats_path)
    return stats, stats_path

  def save_stats(self, model_name):
    stats, _ = self.get_stats_df()

    return stats

  def get_log(self, model_name: str) -> pd.DataFrame:
    _log_fn = f'{model_name}.{self.session_index}.log.csv'
    log_csv = os.path.join(self.model_checkpoint_path, _log_fn)
    try:
      print(f'loading training log from {log_csv}')
      return pd.read_csv(log_csv)
    except:
      print(f'log is not available {log_csv}')

  def get_lr_epoch_from_log(self, model_name) -> (float, int):
    _log = self.get_log(model_name)
    if _log is not None:
      lr = float(_log.iloc[-1]['lr'])
      epoch = int(_log.iloc[-1]['epoch'])
      epoch = max(epoch, len(_log)) + 1
      return lr, epoch
    else:
      return None, 0

  def resave_model_h5(self, model_factory_fn):
    model = self.init_model(model_factory_fn, load_weights=False)
    # model.summary()
    model_name = model_factory_fn.__name__
    ch_fn_old = os.path.join(self.model_checkpoint_path, f"{model_name}.weights")
    model.load_weights(ch_fn_old)
    logger.info(f'model weights loaded: {ch_fn_old}')

    ch_fn = os.path.join(self.model_checkpoint_path, f"{model_name}-{keras.__version__}.h5")

    if not os.path.isfile(ch_fn):
      model.save_weights(ch_fn)
      logger.info(f"model weights saved to {ch_fn}")

    else:
      logger.info(f"model weights NOT saved, because file exists {ch_fn}")

  def init_model(self, model_factory_fn, model_name_override=None, weights_file_override=None,
                 verbose=0,
                 trainable=True,
                 trained=False,
                 load_weights=True,
                 weights=None) -> Model:

    model_name = model_factory_fn.__name__
    if model_name_override is not None:
      model_name = model_name_override

    model = model_factory_fn(name=model_name, ctx=self, trained=trained)
    # model.name = model_name
    if verbose > 1:
      model.summary()


    ch_fn =  self.model_checkpoint_path / f"{model_name}.h5"
    if weights_file_override is not None:
      ch_fn =  self.model_checkpoint_path / f"{weights_file_override}.h5"
    if weights is not None:
      ch_fn = weights

    if load_weights:
      try:
        model.load_weights(ch_fn)
        logger.info(f'weights loaded: {ch_fn}')
      except:
        msg = f'cannot load  {model_name} from {ch_fn}'
        logger.warning(msg)
        # warnings.warn(msg)
        if trained:
          raise FileExistsError(msg)

    if not trainable:
      KerasTrainingContext.freezeModel(model)

    return model

  @staticmethod
  def freezeModel(model):
    model.trainable = False
    for l in model.layers:
      l.trainable = False

  @staticmethod
  def unfreezeModel(model):
    if not model.trainable:
      model.trainable = True
    for l in model.layers:
      l.trainable = True

  def train_and_evaluate_model(self, model:Model, generator, test_generator, retrain=False, lr=None):
    print(f'model.name == {model.name}')
    self.trained_models[model.name] = model.name
    if self.EVALUATE_ONLY:
      print(f'training skipped EVALUATE_ONLY = {self.EVALUATE_ONLY}')
      return

    _log_fn = f'{model.name}.{self.session_index}.log.csv'
    _logger1 = CSVLogger(self.model_checkpoint_path / _log_fn, separator=',', append=not retrain)
    _logger2 = CSVLogger(_log_fn, separator=',', append=not retrain)

    checkpoint_weights = ModelCheckpoint(self.model_checkpoint_path / (model.name + ".h5"),
                                         monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True,
                                         verbose=1)

    lr_logged = None
    if not retrain:
      lr_logged, epoch = self.get_lr_epoch_from_log(model.name)
    else:
      epoch = 0

    if lr_logged is not None:
      K.set_value(model.optimizer.lr, lr_logged)

    if lr is not None:
      K.set_value(model.optimizer.lr, lr)

    print(f'continue: lr:{K.get_value(model.optimizer.lr)}, epoch:{epoch}')


    history = model.fit_generator(
      generator=generator,
      epochs=self.EPOCHS,
      callbacks=[self.reduce_lr, checkpoint_weights, _logger2, _logger1],  # , _logger1, _logger2
      steps_per_epoch=self.steps_per_epoch,
      validation_data=test_generator,
      validation_steps=self.validation_steps,
      initial_epoch=epoch)

    self.HISTORIES[model.name] = history
    self.save_stats(model.name)

    return history

    # plot_training_history(history)
    # plot_compare_models()
