from dotenv import load_dotenv
import os
import traceback
import torch

from configs.base_memory_config import BaseMemoryConfig
from experiment import Experiment


def main():
    try:
        torch.autograd.set_detect_anomaly(True)
        load_dotenv()
        
        experiment = Experiment(config=BaseMemoryConfig)
        experiment.run()

    except Exception as e:
        print("An error occurred during the experiment:", str(e))
        print("Full traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
