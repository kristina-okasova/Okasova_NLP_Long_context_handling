from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    name: str = "meta-llama/Llama-3.2-1B-Instruct"
    thinking_mode: bool = False
    sliding_window_size: Optional[int] = None 
    majority_vote: int = 1
    max_model_len: int = 32768
    multi_turn: bool = True
    cot: bool = False
    
    early_stopping: bool = False
    early_stopping_threshold: float = 0.0