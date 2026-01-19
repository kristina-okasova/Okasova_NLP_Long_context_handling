from dotenv import load_dotenv
import os
import traceback

from configs.infini_attention_config import InfiniAttentionConfig
from configs.infini_armt_config import InfiniARMTConfig
from experiment import Experiment


def main():
    try:
        load_dotenv()

        experiment_name = os.getenv("EXPERIMENT", "ExpA")
        experiment_map = {
            "InfiniAttentionConfig": InfiniAttentionConfig,
            "InfiniARMTConfig": InfiniARMTConfig
        }

        config = experiment_map[experiment_name]()
        
        experiment = Experiment(config=config)
        experiment.run()

    except Exception as e:
        print("An error occurred during the experiment:", str(e))
        print("Full traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()