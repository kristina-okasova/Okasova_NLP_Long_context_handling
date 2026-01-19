# config_definitions.py
from dataclasses import dataclass, field
from ._config import *
from typing import Optional, Literal


# --- Run Settings ---
@dataclass
class RunSettings:
    output_dir: str = "output"
    eval_seed: int = 42
    log_level: str = "INFO"
    model_name: str = ""
    cot: bool = False


# --- Main Configuration Class ---
@dataclass
class MainConfig:
    run_settings: RunSettings = field(default_factory=RunSettings)
    exp: str = "dict_sum" # Default experiment key
    model_config: ModelConfig = field(default_factory=ModelConfig)
    wandb_mode: Optional[Literal["online", "offline", "disabled"]] = "online"
    experiments: Experiments = field(
        default_factory=Experiments
    )
