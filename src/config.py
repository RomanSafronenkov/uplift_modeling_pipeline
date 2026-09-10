import yaml
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Union


class MetricToOptimize(Enum):
    QINI = 'qini'
    UPLIFT_AT_K = 'uplift_at_k'


class GeneralConfig(BaseModel):
    experiment_name: str
    seed: int
    spark_session_name: str
    load_venv_to_spark: bool
    check_target_validity: bool
    create_dataset: bool
    split_data: bool
    select_features: bool
    choose_model: bool
    optimize: bool
    fit_model: bool
    evaluate: bool


class DatasetConfig(BaseModel):
    target_col: str
    date_col: str
    treatment_col: str
    oos_fraction: float


class FeatureSelectionConfig(BaseModel):
    dataset_size: int
    drop_cols: List[str] = Field(default_factory=list)
    num_na_val: Union[int, float]
    string_na_val: str


class OptimizationConfig(BaseModel):
    metric_to_optimize: MetricToOptimize
    k: float = Field(gt=0, le=1)
    use_time_series_split: bool
    n_trials: int
    p_warmup: int


class AppConfig(BaseModel):
    general: GeneralConfig
    dataset: DatasetConfig
    feature_selection: FeatureSelectionConfig
    optimization: OptimizationConfig

_config: Optional[AppConfig] = None


def load_config(yaml_path: Path) -> AppConfig:
    """Load and validate config from yaml"""
    if not yaml_path.exists():
        raise FileNoteFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        raw_data = yaml.safe_load(f)

    try: 
        return AppConfig.model_validate(raw_data)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration in {yaml_path}: {e}")


def get_config(yaml_path: Optional[Path] = None) -> AppConfig:
    """Return config, singleton"""
    global _config
    if _config is None:
        if yaml_path is None:
            yaml_path = Path(__file__).absolute().parent.parent / 'config.yaml'
        _config = load_config(yaml_path)

    return _config

config = get_config()
