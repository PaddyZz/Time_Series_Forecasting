from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    dataset_URL: str
    model_zip_file: Path
    model_unzip_dir: Path
    

@dataclass(frozen=True)
class PretrainModelConfig:
    root_dir: Path

@dataclass(frozen=True)
class TrainModelConfig:
    root_dir: Path
    params_max_epochs: int


@dataclass(frozen=True)
class EvaluateConfig:
    root_dir: Path
    saved_model_dir: Path
    keras_saved_model_dir: Path
    tf_saved_model_dir: Path
    image_saved_dir: Path
    max_subplots:int
    shift:int
    plot_col:str
    input_width: int
    label_width: int
    save_keras: bool
    out_steps: int
    batch_size: int
    plot_origin: bool