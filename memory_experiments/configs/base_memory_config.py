from dataclasses import dataclass

@dataclass
class BaseMemoryConfig:
    model_name : str = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer_name : str = "meta-llama/Llama-3.2-1B-Instruct"
    config_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    dataset_name : str = "emozilla/pg19"
    quantization : bool = False
    lr_scheduler_type : str = "linear"  # based on ARMT
    output_dir : str = "./memory_experiments_tracks"
    with_tracking : bool = True
    report_to : str = "wandb"
    resume_from_checkpoint : str = ""

    seed : int = 42
    block_size : int = 8192  # quarter of the Llama 3.2 1B context length, based on GPU limitations
    batch_size : int = 64  # based on ARMT
    preprocessing_num_workers : int = 1 
    weight_decay : float = 0.01  # do HPO
    learning_rate : float = 1e-5  # based on ARMT
    gradient_accumulation_steps : int = 8  # based on ARMT
    number_of_epochs : int = 5  # do HPO
    num_warmup_steps : int = 100  # do HPO
    max_train_steps : int = 50000  # based on ARMT
    max_valid_steps : int = 5000
    max_test_steps : int = 800
    checkpointing_steps : int = 2000