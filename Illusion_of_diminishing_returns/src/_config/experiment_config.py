from dataclasses import dataclass, field


@dataclass
class BaseExperimentConfig:
    type: str
    num_samples: int = 10
    working_capacity: int = 1
    llm_max_tokens: int = 1000
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    pass_at_k: int = 0


@dataclass
class DictSumExecutionExperimentConfig(BaseExperimentConfig):
    type: str = "DictSum"
    dict_size: int = 10
    horizon_length: int = 20
    min_input_value: int = 1
    max_input_value: int = 15

    # cummulative summary of 0
    chunk_size: int = 5
    reset_cumsum: bool = False

    # dictionary changes
    periodic_change: bool = False
    num_of_changes: int = 0
    change_proportion: float = 0  # proportion how large ratio of positive values is changed
    change_keys: bool = False
    random_dictionary_update: float = 0  # at each step make random choice whether to update dictionary or not with this probability

# --- Top-level Experiments Dictionary ---
@dataclass
class Experiments:
    dict_sum: DictSumExecutionExperimentConfig = field(
        default_factory=DictSumExecutionExperimentConfig
    )