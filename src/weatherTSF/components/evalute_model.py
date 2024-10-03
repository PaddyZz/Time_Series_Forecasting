import tensorflow as tf 
from src.weatherTSF.components.data_windowing import WindowGenerator
from src.weatherTSF.components.single_step_models import compile_and_fit

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

def Baseline_Evaluate(df,train_df,val_df,test_df):
    column_indices = {name: i for i, name in enumerate(df.columns)}
    #baseline = Baseline(label_index=column_indices['T (degC)'])
    baseline = Baseline(label_index=None)
    baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])

    val_performance = {}
    performance = {}
    
    single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=None)
    #label_columns=['T (degC)'])
    
    val_performance['Baseline'] = baseline.evaluate(single_step_window.val, return_dict=True)
    performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0, return_dict=True)
    
    wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=12,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=None)
    #label_columns=['T (degC)'])
    
    print("plot starting")
    
    #model_path = 'src\weatherTSF\models\weatherTSF.keras'
    #loaded_model = tf.keras.models.load_model(model_path)
    wide_window.plot(model=baseline, plot_col='p (mbar)')
    #print(f"{model_path}\n")
    print("plot finished")

def LSTM_Evaluate(df,train_df,val_df,test_df):
    OUT_STEPS=24
    num_features = df.shape[1]

    multi_window = WindowGenerator(
    input_width=24, label_width=24, shift=24,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=None)

    multi_lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units].
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(32, return_sequences=False),
        # Shape => [batch, out_steps*features].
        tf.keras.layers.Dense(OUT_STEPS*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features].
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    _ = compile_and_fit(multi_lstm_model, multi_window)
    multi_lstm_model.save("src\weatherTSF\models\LSTM.keras")
    
    #IPython.display.clear_output()

    _ = multi_lstm_model.evaluate(multi_window.val)
    _ = multi_lstm_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(multi_lstm_model)