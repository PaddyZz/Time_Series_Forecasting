from src.weatherTSF import logger
from src.weatherTSF.components.data_ingestion import DataIngestion
from src.weatherTSF.components.feature_engineering import cleanUpOutlier, splitDataAndNormalization
from src.weatherTSF.components.data_windowing import WindowGenerator
from src.weatherTSF.components.train_model import TrainModel
from src.weatherTSF.components.args import args_cope
from src.weatherTSF.config.configuration import ConfigurationManager
STAGE_NAME ="DATA_INGESTION"
STAGE_NAME_ONE = "PRETRAIN_MODEL"
STAGE_NAME_TWO = "DATA_WINDOWING"
STAGE_NAME_THREE = "TRAIN_MODEL"
STAGE_NAME_FOUR = "MODEL_EVALUATE"

try:

        args_cope()
        data_ingestion_config = ConfigurationManager().get_data_ingestion_config()
        train_model_config  = ConfigurationManager().get_training_config()
        eval_model_config = ConfigurationManager().get_evaluation_config()
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
 
        df, _ = DataIngestion(data_ingestion_config).getDataset()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        logger.info(f">>>>>> stage {STAGE_NAME_ONE} started <<<<<<")
        cleanUpOutlier(df)
        train_df,val_df,test_df,num_features = splitDataAndNormalization(df)
        logger.info(f">>>>>> stage {STAGE_NAME_ONE} completed <<<<<<\n\nx==========x")

        
        logger.info(f">>>>>> stage {STAGE_NAME_TWO} started <<<<<<")
        multi_window = WindowGenerator(config= eval_model_config,train_df=train_df,val_df=val_df,test_df=test_df,df=df)
        
        logger.info(f">>>>>> stage {STAGE_NAME_TWO} completed <<<<<<\n\nx==========x")

        logger.info(f">>>>>> stage {STAGE_NAME_THREE} started <<<<<<")
        TrainModel(multi_window,eval_model_config).LSTM_Train(df)
        logger.info(f">>>>>> stage {STAGE_NAME_THREE} completed <<<<<<\n\nx==========x")
        
        logger.info(f">>>>>> stage {STAGE_NAME_FOUR} started <<<<<<")
        TrainModel(multi_window,eval_model_config).load_and_plot()
        logger.info(f">>>>>> stage {STAGE_NAME_FOUR} completed <<<<<<\n\nx==========x")

        
        
except Exception as e:
        logger.exception(e)
        raise e


