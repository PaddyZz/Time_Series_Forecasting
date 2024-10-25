from src.weatherTSF.constants import *
from src.weatherTSF.utils.common import read_yaml, create_directories
from src.weatherTSF.entity.config_entity import (DataIngestionConfig,
PretrainModelConfig,TrainModelConfig,EvaluateConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            dataset_name=config.dataset_name,
            model_unzip_dir=Path(config.model_unzip_dir),
            dataset_URL=config.dataset_URL,
            model_zip_file=Path(config.model_zip_file)
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PretrainModelConfig:
        config = self.config.pretrain_model
        params = self.params
        create_directories([config.root_dir])

        pretrain_model_config = PretrainModelConfig(
            root_dir=Path(config.root_dir),
        )

        return pretrain_model_config
    



    def get_training_config(self) -> TrainModelConfig:
        config = self.config.train_model
        params = self.params
        create_directories([
            Path(config.root_dir)
        ])

        train_model_config = TrainModelConfig(
            root_dir=Path(config.root_dir),
            params_max_epochs = params.MAX_EPOCHS,
        )

        return train_model_config
    


    def get_evaluation_config(self) -> EvaluateConfig:
        config = self.config.evaluate_model
        params = self.params
        eval_config = EvaluateConfig(
            root_dir=Path(config.root_dir),
            saved_model_dir=Path(config.saved_model_dir),
            keras_saved_model_dir=Path(config.keras_saved_model_dir),
            tf_saved_model_dir = Path(config.tf_saved_model_dir),
            image_saved_dir = Path(config.image_saved_dir),
            max_subplots = params.MAX_SUBPLOTS,
            shift = params.SHIFT,
            plot_col = params.PLOT_COL,
            input_width= params.INPUT_WIDTH,
            label_width= params.LABEL_WIDTH,
            save_keras= params.SAVE_KERAS,
            out_steps= params.OUT_STEPS,
            batch_size= params.BATCH_SIZE,
            plot_origin=params.PLOT_ORIGIN
        )
        return eval_config