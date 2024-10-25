import tensorflow as tf
from src.weatherTSF.config.configuration import (TrainModelConfig)

class CompileAndFit() :
    def __init__(self,config:TrainModelConfig):
        self.config = config
    def compile_and_fit(self, model, window, patience=2):

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            mode='min')

        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])

        
        history = model.fit(window.train, epochs=self.config.params_max_epochs,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        
                            
        return history

