from dataclasses import dataclass

from configs.base_memory_config import BaseMemoryConfig

@dataclass
class InfiniAttentionConfig(BaseMemoryConfig):
    memory_approach : str = "infini_attention"

    # Memory approach specific parameters
    segment_length : int = 512  # based on ARMT
