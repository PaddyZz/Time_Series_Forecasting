from src.weatherTSF import logger

import tensorflow as tf
from src.weatherTSF.components.data_injection import getDataset
from src.weatherTSF.components.pretrian_model import cleanUpOutlier, splitDataAndNormalization
from src.weatherTSF.components.data_windowing import WindowGenerator,split_window,make_dataset
from src.weatherTSF.components.single_step_models import compile_and_fit
import mlflow

STAGE_NAME ="DATA_INGESTION"
STAGE_NAME_ONE = "PRETRAIN_MODEL"
STAGE_NAME_TWO = "DATA_WINDOWING"
STAGE_NAME_THREE = "SINGLE_STEP_MODELS"

try:

        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        #mlflow.set_tracking_uri("https://dagshub.com/PaddyZz/TimeSeiresForcasting-Weather.mlflow")
        #mlflow.set_experiment("weatherTSF")
        #mlflow.log_param("learning_rate", 0.01)
        df, date_time = getDataset()
        print(df.head())
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        logger.info(f">>>>>> stage {STAGE_NAME_ONE} started <<<<<<")
        column_indices = {name: i for i, name in enumerate(df.columns)}
        cleanUpOutlier(df)
        train_df,val_df,test_df,num_features = splitDataAndNormalization(df)
        logger.info(f">>>>>> stage {STAGE_NAME_ONE} completed <<<<<<\n\nx==========x")

        logger.info(f">>>>>> stage {STAGE_NAME_TWO} started <<<<<<")
        WindowGenerator.split_window = split_window
        WindowGenerator.make_dataset = make_dataset
        wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=1,
        train_df=train_df,val_df=val_df,test_df=test_df,
        label_columns=['T (degC)'])
        logger.info(f">>>>>> stage {STAGE_NAME_TWO} completed <<<<<<\n\nx==========x")

        logger.info(f">>>>>> stage {STAGE_NAME_THREE} started <<<<<<")
        lstm_model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(32, return_sequences=True),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=1)
        ])
        history = compile_and_fit(lstm_model, wide_window)
        lstm_model.save("./src/weatherTSF/models/lstm/weatherTSF.keras")
        loaded_model = tf.keras.models.load_model('./src/weatherTSF/models/lstm/weatherTSF.keras')
        print(loaded_model.summary())
        #with mlflow.start_run():
         #       mlflow.tensorflow.log_model(loaded_model, "lstm_model")
        logger.info(f">>>>>> stage {STAGE_NAME_THREE} completed <<<<<<\n\nx==========x")
        
except Exception as e:
        logger.exception(e)
        raise e


