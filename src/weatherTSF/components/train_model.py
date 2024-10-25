import tensorflow as tf 
from src.weatherTSF.components.compile_and_fit import CompileAndFit
from src.weatherTSF.config.configuration import (EvaluateConfig)
from src.weatherTSF.config.configuration import ConfigurationManager

class TrainModel():
  def __init__(self,multi_window,config:EvaluateConfig):
     self.config = config
     self.multi_window =multi_window
     self.train_model_config = ConfigurationManager().get_training_config()
  def LSTM_Train(self,df):
      BATCH_SIZE = self.config.batch_size
      OUT_STEPS= self.config.out_steps
      num_features = df.shape[1]
      keras_saved_model_sign = self.config.save_keras
      multi_lstm_model = tf.keras.Sequential([

          tf.keras.layers.LSTM(BATCH_SIZE, return_sequences=False),

          tf.keras.layers.Dense(OUT_STEPS*num_features,
                              kernel_initializer=tf.initializers.zeros()),
          tf.keras.layers.Reshape([OUT_STEPS, num_features])
      ])

      _ = CompileAndFit(self.train_model_config).compile_and_fit(multi_lstm_model, self.multi_window)
      if keras_saved_model_sign:
        multi_lstm_model.save(self.config.keras_saved_model_dir)
      else: 
        tf.saved_model.save(multi_lstm_model,self.config.tf_saved_model_dir)

  def load_and_plot(self):
    # load model and plot image
    loaded_model = None
    if self.config.save_keras:
            loaded_model = tf.keras.models.load_model(self.config.keras_saved_model_dir )
    else:
            loaded_model = tf.saved_model.load(self.config.tf_saved_model_dir)
    if loaded_model is not None:
            self.multi_window.plot(model=loaded_model)
