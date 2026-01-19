import os
import datetime
import json
import logging
import wandb
from dotenv import load_dotenv

# Project-specific imports
from src._config import ModelConfig, Experiments
from src.config import MainConfig, RunSettings
from src.experiments import (
    DictSumExecutionExperiment,
)

# Assuming you have a unified LLM client in llm_clients.py
from llm_clients import UnifiedLLM
from utils import save_json_to_file  # Assuming this utility function exists


class ExperimentRunner:
    def __init__(self, config: MainConfig):
        self.config = config

        # Access attributes directly from the MainConfig object
        self.run_settings: RunSettings = self.config.run_settings
        self.model_config_params: ModelConfig = self.config.model_config
        self.exp = self.config.exp

        if not self.exp:
            raise ValueError("No 'exp' found in config.")
        
        self.output_dir_base = self.run_settings.output_dir
        self.run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_name_for_path = self.model_config_params.name.split("/")[-1]
        
        self.run_output_dir = os.path.join(
            self.output_dir_base,
            f"{self.exp}/{self.run_timestamp}_{self.model_name_for_path}",
        )

        os.makedirs(self.run_output_dir, exist_ok=True)
        logging.info(f"Results for this run will be saved in: {self.run_output_dir}")

        self.thinking_mode = self.model_config_params.thinking_mode
        self.is_chat_mode = self.model_config_params.multi_turn
        self.is_cot = self.model_config_params.cot
        
        if self.is_chat_mode:
            logging.info("Chat mode is enabled for this run.")
        else:
            logging.info("Chat mode is disabled for this run.")

        if self.is_cot:
            logging.info("Chain-of-Thought (CoT) mode is enabled for this run.")
        else:
            logging.info("Chain-of-Thought (CoT) mode is disabled for this run.")

        self.run_settings.model_name = self.model_config_params.name

        experiment_configs_all: Experiments = self.config.experiments
        self.current_experiment_config = getattr(experiment_configs_all, self.exp)
        
        # add cot to experiment config if not present
        if not hasattr(self.run_settings, 'cot'):
            self.run_settings.cot = self.is_cot

        if not self.current_experiment_config:
            raise ValueError(
                f"Configuration for active experiment '{self.exp}' not found in 'experiments' section."
            )

        logging.info(
            f"Thinking mode is {'enabled' if self.thinking_mode else 'disabled'} for this run."
        )

        self.wandb_run = None
        self._init_wandb()
        self._setup_logging()  # Logging setup should happen before potential wandb init logging

        self.llm_client = None
        self.experiment_instance = None

        self.NOOP_TOKEN = "<NoOp>"
        self.INVALID_FORMAT_TOKEN = "<IncorrectFormat>"

    def _setup_logging(self):
        """Configures logging based on config."""
        log_level = self.run_settings.log_level.upper()
        if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            log_level = logging.INFO
        else:
            log_level = getattr(logging, log_level)

        # detach from any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            handler.close()

        self.log_path = os.path.join(self.run_output_dir, "experiment.log")

        logging.basicConfig(
            filename=self.log_path,
            level=log_level,
            datefmt="%Y-%m-%d %H:%M:%S",
            format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            force=True,  # Force reconfiguration of logging
        )

        logging.info(f"Setting up logging with level: {log_level} for run: {self.exp}")

    def _init_llm_client(self):
        """Initializes the UnifiedLLM client."""
        logging.info("Initializing LLM client...")
        try:
            self.llm_client = UnifiedLLM(
                model_name=self.model_config_params.name,
                model_config=self.model_config_params
            )
            logging.info(
                f"LLM Client initialized: {self.llm_client.get_model_name()}"
            )
        except Exception as e:
            logging.error(f"Failed to initialize LLM client: {e}", exc_info=True)
            raise

    def _init_experiment_instance(self):
        """Initializes the specific experiment class instance using a factory pattern."""
        # Accessing dataclass attributes directly
        experiment_type_name = self.current_experiment_config.type
        if not experiment_type_name:
            raise ValueError(f"Experiment type not specified for '{self.exp}'.")

        logging.info(f"Initializing experiment type: {experiment_type_name}")
        try:
            experiment_class_map = {
                "DictSum": DictSumExecutionExperiment,
            }

            if experiment_type_name in experiment_class_map:
                ExperimentClass = experiment_class_map[experiment_type_name]
            else:
                raise ValueError(
                    f"Unknown experiment type: {experiment_type_name}. Please check your config."
                )

            self.experiment_instance = ExperimentClass(
                common_config=self.run_settings,  # Pass dataclass directly
                experiment_specific_config=self.current_experiment_config,  # Pass dataclass directly
            )

            # Set additional naming information for plotting and logging
            self.experiment_instance.model_name_for_path = self.model_name_for_path
            self.experiment_instance.run_timestamp = self.run_timestamp
            self.experiment_instance.is_multi_turn = self.is_chat_mode
            logging.info(
                f"Experiment instance initialized: {self.experiment_instance.__class__.__name__}"
            )

        except Exception as e:
            logging.error(
                f"Error initializing experiment instance {experiment_type_name}: {e}",
                exc_info=True,
            )
            raise


    def _multi_turn_run(self, prompts, sampling_params):
        """Handles multi-turn prompt generation and LLM interaction."""
        logging.info("Running in multi-turn mode...")
        if not self.is_chat_mode:
            raise ValueError("Multi-turn mode is enabled but not configured correctly.")

        llm_raw_outputs = None
        if self.experiment_instance and self.llm_client:
            start_template = self.experiment_instance.get_start_template()

            llm_raw_outputs = self.llm_client.chat_generate_step_wise(
                user_prompts_list=prompts,
                start_template=start_template if start_template else "",
                enable_thinking=self.thinking_mode,
                evaluator=self.experiment_instance.evaluator,
                **sampling_params,
            )
        return llm_raw_outputs

    def _single_turn_run(self, prompts, sampling_params):
        """Handles single-turn prompt generation and LLM interaction."""
        logging.info("Running in single-turn mode...")
        if self.is_chat_mode:
            raise ValueError(
                "Single-turn mode is enabled but multi-turn is configured."
            )
        # THINK_SUFFX = ""
        # if "qwen" in self.llm_client.model_name.lower():
        #     THINK_SUFFX = "/think" if self.thinking_mode else "/no_think"
        # prompts_with_suffix = [prompt + THINK_SUFFX for prompt in prompts]
        llm_raw_outputs = None
        if self.llm_client:
            llm_raw_outputs = self.llm_client.generate(
                prompts=prompts, enable_thinking=self.thinking_mode, **sampling_params
            )
        return llm_raw_outputs

    def _init_wandb(self):
        """Initializes a new W&B run if enabled."""
        # Accessing dataclass attributes directly
        if self.config.wandb_mode == "disabled":
            logging.info("W&B logging is disabled.")
            return

        try:
            load_dotenv()
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            # Redact sensitive info before logging
            safe_model_config_params = self.model_config_params
            # Make a deep copy to modify without affecting original dataclass
            safe_model_config_params = json.loads(
                json.dumps(vars(safe_model_config_params))
            )  # Simple deep copy

            # Redact sensitive info from the dicts
            keys_to_redact = ["api_key", "apikey", "api-key"]

            def redact_dict(d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        d[k] = redact_dict(v)
                    elif any(redact_key in k.lower() for redact_key in keys_to_redact):
                        d[k] = "REDACTED"
                return d

            safe_model_config_params = redact_dict(safe_model_config_params)

            wandb_init_config = {
                "run_info": {
                    "experiment_key": self.exp,
                    "run_timestamp": self.run_timestamp,
                    "output_directory": self.run_output_dir
                },
                "common_settings_used": vars(self.run_settings),  # Convert to dict
                "model_parameters_used": safe_model_config_params,
                "experiment_parameters_used": vars(self.current_experiment_config)
            }

            self.wandb_run = wandb.init(
                config=wandb_init_config,
                dir=self.run_output_dir,
                mode=self.config.wandb_mode,
                job_type="experiment_run",
            )
            logging.info(
                f"W&B run initialized: {self.wandb_run.url if self.wandb_run and hasattr(self.wandb_run, 'url') else 'Mode: ' + str(self.config.wandb_mode)}"
            )

        except Exception as e:
            logging.error(f"Failed to initialize W&B: {e}", exc_info=True)
            self.wandb_run = None

    def _calculate_invalids(self, parsed_llm_outputs):
        """
        Calculates the average number of NoOps and invalid format tokens at each step.

        This optimized version combines counting and formatting into a single loop
        to improve efficiency and readability.

        Args:
            parsed_llm_outputs: A list of sequences, where each sequence is a list of tokens.

        Returns:
            A tuple containing two lists of dictionaries:
            - The first list contains the average count of NoOps at each step.
            - The second list contains the average count of invalids at each step.
        """
        num_steps = len(parsed_llm_outputs[0]) if parsed_llm_outputs else 0
        if num_steps == 0:
            logging.warning("No steps found in LLM raw outputs. Returning empty lists.")
            return [], []
        num_sequences = len(parsed_llm_outputs)

        if not num_sequences:
            # Return empty lists with the correct structure if input is empty
            return [{"step": i, "no_ops": 0.0} for i in range(num_steps)], [
                {"step": i, "invalids": 0.0} for i in range(num_steps)
            ]

        no_ops_counts = [0] * num_steps
        invalids_counts = [0] * num_steps

        # Single pass to count tokens
        for output_sequence in parsed_llm_outputs:
            for step_idx, token in enumerate(output_sequence[:num_steps]):
                if token == self.NOOP_TOKEN:
                    no_ops_counts[step_idx] += 1
                elif token == self.INVALID_FORMAT_TOKEN:
                    invalids_counts[step_idx] += 1

        # Use list comprehensions for a concise and direct creation of the final structure
        no_ops_list = [
            {"step": i, "no_ops": count / num_sequences}
            for i, count in enumerate(no_ops_counts)
        ]
        invalids_list = [
            {"step": i, "invalids": count / num_sequences}
            for i, count in enumerate(invalids_counts)
        ]

        return no_ops_list, invalids_list

    def run(self):
        """Orchestrates the full experiment lifecycle."""
        try:
            logging.info(f"--- Starting Experiment Run: {self.exp} ---")
            logging.info(
                f"Model: {self.model_config_params.name}"
            )

            # 1. Initialize LLM Client
            self._init_llm_client()
            if not self.llm_client:
                if self.wandb_run:
                    wandb.summary["status"] = "LLM Client Init Failed"
                return

            # 2. Initialize Experiment Instance
            self._init_experiment_instance()
            if not self.experiment_instance:
                if self.wandb_run:
                    wandb.summary["status"] = "Experiment Init Failed"
                return
            
            # NEW FOR DICT SUM
            self.experiment_instance.evaluator.set_wandb_logger(self.wandb_run)
            
            logging.info("Experiment setup complete.")

            # 3. Prepare Prompts
            logging.info("Preparing prompts...")

            prompts = self.experiment_instance.prepare_prompts()

            self.custom_inference(prompts)

            llm_raw_outputs = list()
            if not prompts:
                logging.warning("No prompts were generated. Skipping LLM generation.")
            else:
                logging.info(f"Prepared {len(prompts)} prompts.")
                if prompts:
                    logging.debug(
                        f"Example prompt (first one):\n{str(prompts[0])}"
                    )

                sampling_params = {
                    "max_tokens": self.current_experiment_config.llm_max_tokens,
                    "temperature": self.current_experiment_config.llm_temperature,
                    "top_p": self.current_experiment_config.llm_top_p,
                    "majority_vote": self.model_config_params.majority_vote,
                }

                logging.info(f"LLM Sampling Parameters: {sampling_params}")

                # 4. Run LLM Generation
                if self.is_chat_mode:
                    llm_raw_outputs = self._multi_turn_run(prompts, sampling_params)
                else:
                    llm_raw_outputs = self._single_turn_run(prompts, sampling_params)
                    
                if llm_raw_outputs:
                    logging.info(
                        f"LLM generation complete. Received {len(llm_raw_outputs)} outputs."
                    )
                    logging.debug(
                        f"Example LLM output (first one):\n{str(llm_raw_outputs[0])}..."
                    )

            # 5. Process LLM Outputs
            logging.info("Processing LLM outputs...")
            if llm_raw_outputs:
                if self.is_chat_mode:
                    processed_outputs = self.experiment_instance.process_llm_outputs(
                        llm_raw_outputs, enable_thinking=self.thinking_mode
                    )
                    if processed_outputs:
                        logging.info(f"Processed {len(processed_outputs)} LLM outputs.")
                else:
                    processed_outputs = self.experiment_instance.process_llm_outputs(llm_raw_outputs, enable_thinking=self.thinking_mode)

            # 6. Evaluate Predictions
            logging.info("Evaluating predictions...")
            evaluation_results = self.experiment_instance.evaluate_predictions()
            logging.info("Evaluation complete.")
            aggregated_metrics = evaluation_results.get("aggregated_metrics", {})
            logging.info(f"Aggregated Evaluation Metrics: {aggregated_metrics}")

            no_ops_list, invalids_list = self._calculate_invalids(
                self.experiment_instance.parsed_llm_outputs
            )

            aggregated_metrics["no_ops_per_step"] = no_ops_list
            aggregated_metrics["invalids_per_step"] = invalids_list
            
            # 8. Save Results
            logging.info("Saving results...")
            results_filepath = self._save_all_results(evaluation_results)

            if (
                self.wandb_run
                and results_filepath
            ):
                # log metrics to wandb
                
                # Get bootstrap confidence intervals if available
                bootstrap_ci = evaluation_results.get("bootstrap_confidence_intervals", {})
                        
                for step in range(len(aggregated_metrics.get("avg_prefix_accuracy_per_position", []))):
                    prefix_arr = aggregated_metrics.get("avg_prefix_accuracy_per_position", [])
                    index_arr = aggregated_metrics.get("avg_index_accuracy_per_position", [])
                    step_arr = aggregated_metrics.get("avg_step_accuracy_per_position", [])
                    no_ops = aggregated_metrics.get("no_ops_per_step", [])
                    invalids = aggregated_metrics.get("invalids_per_step", [])
                    
                    metrics_to_log = {
                        f"prefix_accuracy": prefix_arr[step],
                        f"index_accuracy": index_arr[step],
                        f"step_accuracy": step_arr[step],
                        f"no_ops": no_ops[step],
                        f"invalids": invalids[step],
                    }
                    
                    # Add bootstrap confidence intervals if available
                    if bootstrap_ci:
                        if "prefix_accuracy_ci" in bootstrap_ci and step < len(bootstrap_ci["prefix_accuracy_ci"]["lower"]):
                            metrics_to_log.update({
                                f"prefix_accuracy_ci_lower": bootstrap_ci["prefix_accuracy_ci"]["lower"][step],
                                f"prefix_accuracy_ci_upper": bootstrap_ci["prefix_accuracy_ci"]["upper"][step],
                            })
                        
                        if "step_accuracy_ci" in bootstrap_ci and step < len(bootstrap_ci["step_accuracy_ci"]["lower"]):
                            metrics_to_log.update({
                                f"step_accuracy_ci_lower": bootstrap_ci["step_accuracy_ci"]["lower"][step],
                                f"step_accuracy_ci_upper": bootstrap_ci["step_accuracy_ci"]["upper"][step],
                            })
                        
                        if "index_accuracy_ci" in bootstrap_ci and step < len(bootstrap_ci["index_accuracy_ci"]["lower"]):
                            metrics_to_log.update({
                                f"index_accuracy_ci_lower": bootstrap_ci["index_accuracy_ci"]["lower"][step],
                                f"index_accuracy_ci_upper": bootstrap_ci["index_accuracy_ci"]["upper"][step],
                            })
                    
                    if not self.is_chat_mode:
                        self.wandb_run.log(metrics_to_log, step=step)
                        # only log for single-turn mode
                    
                    # Enhanced logging with confidence intervals
                    if bootstrap_ci:
                        ci_info = ""
                        if "prefix_accuracy_ci" in bootstrap_ci and step < len(bootstrap_ci["prefix_accuracy_ci"]["lower"]):
                            ci_info += f" [CI: {bootstrap_ci['prefix_accuracy_ci']['lower'][step]:.3f}-{bootstrap_ci['prefix_accuracy_ci']['upper'][step]:.3f}]"
                        logging.info(
                            f"Step {step} - "
                            f"Prefix Accuracy: {prefix_arr[step]:.3f}{ci_info}, "
                            f"Index Accuracy: {index_arr[step]:.3f}, "
                            f"Step Accuracy: {step_arr[step]:.3f}, "
                            f"No Ops: {no_ops[step]}, "
                            f"Invalids: {invalids[step]}"
                        )
                    else:
                        logging.info(
                            f"Step {step} - "
                            f"Prefix Accuracy: {prefix_arr[step]}, "
                            f"Index Accuracy: {index_arr[step]}, "
                            f"Step Accuracy: {step_arr[step]}, "
                            f"No Ops: {no_ops[step]}, "
                            f"Invalids: {invalids[step]}"
                        )
                
                try:
                    multi_turn_suffix = (
                        "multi-turn" if self.is_chat_mode else "single-turn"
                    )
                    artifact_name = f"{self.run_timestamp}_{self.model_name_for_path}_{self.exp}_{multi_turn_suffix}_results"
                    artifact = wandb.Artifact(
                        name=artifact_name,
                        type="experiment_results",
                        description=f"Results for {self.exp} run {self.run_timestamp} with {self.model_name_for_path} ({'multi-turn' if self.is_chat_mode else 'single-turn'})",
                        metadata={
                            "experiment_key": self.exp,
                            "timestamp": self.run_timestamp,
                            "model": self.model_config_params.name,
                            "model_name_for_path": self.model_name_for_path,
                            "is_multi_turn": self.is_chat_mode,
                        },
                    )
                    artifact.add_file(results_filepath)
                    artifact.add_file(self.log_path)
                    self.wandb_run.log_artifact(artifact)
                    logging.info(
                        f"Logged results JSON as W&B artifact: {artifact_name}"
                    )
                except Exception as e:
                    logging.error(f"Failed to log W&B artifact: {e}", exc_info=True)

            # 8. Run Experiment-Specific Plotting
            logging.info("Attempting experiment-specific plotting...")
            multi_turn_suffix = "multi-turn" if self.is_chat_mode else "single-turn"
            base_plot_filename = f"{self.run_timestamp}_{self.model_name_for_path}_{self.exp}_{multi_turn_suffix}_plots"

            self.experiment_instance.run_plotting(
                self.run_output_dir, base_plot_filename
            )

            logging.info(f"--- Experiment Run {self.exp} Finished ---")
            logging.info(f"All results saved in: {self.run_output_dir}")

        except Exception as e:
            logging.error(
                f"An error occurred during the experiment run: {e}", exc_info=True
            )
            if self.wandb_run:
                wandb.summary["status"] = "Crashed"
                wandb.summary["error_message"] = str(e)
            raise
        finally:
            if self.wandb_run and wandb.run is not None:
                exit_code = 0
                if wandb.summary.get("status") == "Crashed":
                    exit_code = 1
                elif (
                    wandb.summary.get("status") == "LLM Client Init Failed"
                    or wandb.summary.get("status") == "Experiment Init Failed"
                ):
                    exit_code = 2

                self.wandb_run.finish(exit_code=exit_code)
                logging.info(f"W&B run finished with exit code {exit_code}.")
            self.wandb_run = None

    def _convert_lists_to_strings(self, data):
        """
        Recursively traverses a dictionary or list and converts any list found into a string.

        Args:
            data: The dictionary or data structure to traverse.

        Returns:
            A new data structure with all lists converted to strings.
        """
        if isinstance(data, dict):
            # If it's a dictionary, recurse into its values.
            return {
                key: self._convert_lists_to_strings(value)
                for key, value in data.items()
            }

        elif isinstance(data, list):
            # --- NEW LOGIC IS HERE ---
            # Check if the list is non-empty and if ALL of its elements are lists.
            is_list_of_lists = bool(data) and all(
                isinstance(item, list) for item in data
            )

            is_list_of_dicts = bool(data) and all(
                isinstance(item, dict) for item in data
            )

            if is_list_of_lists or is_list_of_dicts:
                # If it IS a list of lists, DO NOT convert it to a string.
                # Instead, recurse into each sub-list to process them further.
                return [self._convert_lists_to_strings(sub_list) for sub_list in data]
            else:
                # If it's a normal list (containing primitives, dicts, mixed types)
                # or an empty list, convert it to a string.
                return str(data)
        else:
            # For any other data type (string, int, etc.), return it as is.
            return data

    def _save_all_results(self, evaluation_results):
        """Consolidates and saves all results to a JSON file and returns the filepath."""
        exp_params_to_log = dict()
        if self.experiment_instance:
            exp_params_to_log = self.experiment_instance.get_experiment_params_for_logging()

        model_params_to_log = vars(self.model_config_params)

        # Add actual LLM client info if available
        if self.llm_client:
            model_params_to_log["model_name"] = self.llm_client.model_name # type: ignore

        # remove sensitive info
        keys_to_redact = [
            "api_key",
            "apikey",
            "api-key",
        ]

        def redact_sensitive_info(params_dict):
            # Operate on a copy to avoid modifying original dict during iteration if not careful
            safe_dict = {}
            for k, v in params_dict.items():
                if isinstance(v, dict):
                    safe_dict[k] = redact_sensitive_info(v)
                elif isinstance(v, list) and all(isinstance(item, dict) for item in v):
                    safe_dict[k] = [redact_sensitive_info(item) for item in v]
                elif any(redact_key in k.lower() for redact_key in keys_to_redact):
                    safe_dict[k] = "REDACTED"
                else:
                    safe_dict[k] = v
            return safe_dict

        model_params_to_log = redact_sensitive_info(model_params_to_log)

        final_results_data = {
            "run_info": {
                "experiment_key": self.exp,
                "run_timestamp": self.run_timestamp,
                "output_directory": self.run_output_dir,
            },
            "common_settings_used": self.run_settings,  # Convert to dict
            "model_parameters_used": model_params_to_log,  # Already a dict
            "experiment_parameters_used": dict(
                exp_params_to_log
            ),  # Convert to dict (assuming it returns dataclass)
            "evaluation_summary": evaluation_results.get(
                "aggregated_metrics", {}
            ),  # Assuming evaluation_results is a dict
            "detailed_experiment_outputs": self.experiment_instance.get_results_to_save() if self.experiment_instance else None,
            "extreme_values": self.experiment_instance.generator_util.extreme_values if self.experiment_instance is not None else list()
        }

        # convert lists to strings in the final results data
        final_results_data = self._convert_lists_to_strings(final_results_data)

        results_filepath = os.path.join(self.run_output_dir, "experiment_results.json")
        try:
            save_json_to_file(final_results_data, results_filepath)
            logging.info(f"All results saved to: {results_filepath}")
        except Exception as e:
            logging.error(
                f"Failed to save results JSON to {results_filepath}: {e}", exc_info=True
            )
            return None
        return results_filepath



