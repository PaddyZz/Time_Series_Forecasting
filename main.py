from src.weatherTSF import logger
import sys
import tensorflow as tf
from src.weatherTSF.components.data_ingestion import getDataset
from src.weatherTSF.components.pretrian_model import cleanUpOutlier, splitDataAndNormalization
from src.weatherTSF.components.data_windowing import WindowGenerator
from src.weatherTSF.components.single_step_models import compile_and_fit
from src.weatherTSF.components.evalute_model import LSTM_Evaluate
from src.weatherTSF.components.autoreg_train_model import FeedBack
from src.weatherTSF.components.autoreg_eval_models import autoregressive_LSTM_Evaluate

STAGE_NAME ="DATA_INGESTION"
STAGE_NAME_ONE = "PRETRAIN_MODEL"
STAGE_NAME_TWO = "DATA_WINDOWING"
STAGE_NAME_THREE = "SINGLE_STEP_MODELS"
STAGE_NAME_FOUR = "MODEL_EVALUATE"

try:

        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        df, date_time = getDataset()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        logger.info(f">>>>>> stage {STAGE_NAME_ONE} started <<<<<<")
        column_indices = {name: i for i, name in enumerate(df.columns)}
        cleanUpOutlier(df)
        train_df,val_df,test_df,num_features = splitDataAndNormalization(df)
        logger.info(f">>>>>> stage {STAGE_NAME_ONE} completed <<<<<<\n\nx==========x")

        
        logger.info(f">>>>>> stage {STAGE_NAME_TWO} started <<<<<<")
        #WindowGenerator.split_window = split_window
        #WindowGenerator.make_dataset = make_dataset
        wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=24,
        train_df=train_df,val_df=val_df,test_df=test_df,
        label_columns=None)
        """
        logger.info(f">>>>>> stage {STAGE_NAME_TWO} completed <<<<<<\n\nx==========x")

        logger.info(f">>>>>> stage {STAGE_NAME_THREE} started <<<<<<")
        lstm_model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(32, return_sequences=True),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=1)
        ])
        history = compile_and_fit(lstm_model, wide_window)

        if 'google.colab' in sys.modules:
                model_path = '/content/TimeSeiresForcasting-Weather/src/weatherTSF/models/weatherTSF.keras'
        else:
                model_path = './src/weatherTSF/models/weatherTSF.keras'
        
        lstm_model.save(model_path)
        loaded_model = tf.keras.models.load_model(model_path)
        print(loaded_model.summary())
        logger.info(f">>>>>> stage {STAGE_NAME_THREE} completed <<<<<<\n\nx==========x")
        """
        logger.info(f">>>>>> stage {STAGE_NAME_FOUR} started <<<<<<")
        
        #LSTM_Evaluate(df,train_df=train_df,val_df=val_df,test_df=test_df)
        
        loaded_model = tf.saved_model.load('./src/weatherTSF/models/lstm/')
        wide_window.plot(model=loaded_model, saveModelSign = True)

        logger.info(f">>>>>> stage {STAGE_NAME_FOUR} completed <<<<<<\n\nx==========x")

        
        
except Exception as e:
        logger.exception(e)
        raise e


