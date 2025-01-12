import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.weatherTSF.config.configuration import (EvaluateConfig)

class WindowGenerator():
  def __init__(self,config:EvaluateConfig,train_df, val_df, test_df, df):
    self.config = config
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.df = df
    # Work out the label column indices.

    self.label_columns =  [self.config.plot_col] 
    if self.label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(self.label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    input_width = self.config.input_width
    label_width = self.config.label_width
    shift = self.config.shift
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result

  def plot(self, model=None,  max_subplots=1):
    inputs, labels = self.example
    saveModelSign = self.config.save_keras
    plot_col=self.config.plot_col
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index
      if self.config.plot_origin:
        plt.ylabel(f'{plot_col}')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index]*self.df[self.config.plot_col].std() + self.df[self.config.plot_col].mean(),
                label='Inputs', marker='.', zorder=-10)
        plt.scatter(self.label_indices, labels[n, :, label_col_index]*self.df[self.config.plot_col].std() + self.df[self.config.plot_col].mean(),
                  edgecolors='k', label='Labels', c='#2ca02c', s=64)
      else:
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                label='Inputs', marker='.', zorder=-10)
        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                  edgecolors='k', label='Labels', c='#2ca02c', s=64)
      if (model is not None) and (saveModelSign is True):
        predictions = model(inputs)
        if self.config.plot_origin:
          plt.scatter(self.label_indices, predictions[n, :, label_col_index]*self.df[self.config.plot_col].std() + self.df[self.config.plot_col].mean(),
                      marker='X', edgecolors='k', label='Predictions',
                      c='#ff7f0e', s=64)
        else:
          plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                      marker='X', edgecolors='k', label='Predictions',
                      c='#ff7f0e', s=64)
      if (model is not None) and (saveModelSign is False):
        infer = model.signatures['serving_default']
        predictions = infer(inputs)
        pred_tensor = predictions['output_0']
        if self.config.plot_origin:
          plt.scatter(self.label_indices, pred_tensor[n, :, label_col_index ]*self.df[self.config.plot_col].std() + self.df[self.config.plot_col].mean(),
              marker='X', edgecolors='k', label='Predictions',
              c='#ff7f0e', s=64)
        else:
          plt.scatter(self.label_indices, pred_tensor[n, :, label_col_index ],
                      marker='X', edgecolors='k', label='Predictions',
                      c='#ff7f0e', s=64)

      if n == 0:
        plt.legend()
    plt.savefig(self.config.image_saved_dir,dpi=500, bbox_inches='tight')
    plt.xlabel('Time [h]')
    plt.show()

  def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

  def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds