# llm_clients.py
import random
import time
import torch
import openai  # For OpenRouter
from tqdm import tqdm
from vllm import (
    LLM as VLLM_Engine,
    SamplingParams as VLLM_SamplingParams,
)  # Rename to avoid class name clash
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
import logging  # Or use standard logging
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter
import logging as std_logging
import functools

from src._config.model_config import ModelConfig
import re

std_logging.getLogger("openai").setLevel(std_logging.WARNING)
std_logging.getLogger("httpx").setLevel(std_logging.WARNING)


def timing_decorator(func):
    """
    Decorator to measure and log the execution time of a function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Try to get a meaningful name for logging
        if hasattr(func, "__self__"):
            class_name = func.__self__.__class__.__name__
            func_name = f"{class_name}.{func.__name__}"
        else:
            func_name = func.__name__

        logging.info(f"⏱️  {func_name} completed in {execution_time:.4f} seconds")
        return result

    return wrapper


# Helper function to load the actual client (internal to this module)
def _load_actual_client(
    model_name: str,
    model_config: Dict[str, Any]
) -> VLLM_Engine | None:
    """
    Loads the appropriate LLM client instance based on the provider.
    This is a helper function for the UnifiedLLM class.
    """
    logging.info(f"Attempting to load vLLM model: {model_name}...")
    max_model_len = model_config.get("max_model_len")
    dtype = model_config.get("dtype", "auto")
    enforce_eager = model_config.get(
        "enforce_eager", False
    )  # Example of another vLLM param

    try:
        client = VLLM_Engine(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.9,
            dtype=dtype,
            enforce_eager=enforce_eager,
        )
        logging.info(f"vLLM model loaded successfully: {model_name}")
        return client
    except Exception as e:
        logging.error(
            f"Error loading vLLM model ({model_name}): {e}", exc_info=True
        )
        logging.error(
            "Please ensure vLLM is installed correctly, the model is accessible, "
            "and GPU resources are available and compatible."
        )
        return None

class UnifiedLLM:
    """
    A unified wrapper class for interacting with different LLM providers (vLLM, OpenRouter).
    """

    def __init__(
        self,
        model_name: str,
        model_config: ModelConfig
    ):
        self.model_name = model_name
        self.model_config = model_config

        self.sliding_window_size = model_config.sliding_window_size
        self.early_stopping = model_config.early_stopping
        self.early_stopping_threshold = model_config.early_stopping_threshold

        # self.openrouter_config = openrouter_config
        self.client = _load_actual_client(
            model_name, model_config # type: ignore
        )

        if self.client is None:
            raise ValueError(
                f"Failed to load LLM client for provider vLLM and model {model_name}."
            )

        logging.info(
            f"UnifiedLLM initialized for provider vLLM, model: {self.model_name}"
        )
        self.tokens_generated_per_step: List[int] = []

    def _remove_thinking_traces(self, message: str) -> str:
        """
        Remove thinking traces and extract clean content from LLM responses.

        First tries to remove <think>...</think> tags while preserving other content.
        If no thinking tags found, looks for <answer>...</answer> tags.
        If neither found, returns empty string.
        """
        thinking_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
        answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

        # Try to remove thinking traces first
        if thinking_pattern.search(message):
            cleaned_message = thinking_pattern.sub("", message).strip()
            logging.debug("Removed thinking traces from the message.")
        else:
            cleaned_message = message.strip()
            # Look for answer tags if no thinking traces found
            answer_match = answer_pattern.search(message)
            if answer_match:
                content = answer_match.group(1).strip()
                if self.model_config.cot:
                    cleaned_message = message  # give cot trace to history
                else:
                    cleaned_message = "<answer>" + content + "</answer>"
                logging.debug("Found CoT answer in the message")
            else:
                cleaned_message = "No Answer"
                logging.warning(
                    f"No thinking traces or CoT answer found in the message: {message}..."
                )

        logging.debug(f"Cleaning during history processing: {cleaned_message}")
        return cleaned_message

    def _process_single_instance_vllm_chat(
        self,
        instance_idx: int,
        instance_prompts: List[str],
        start_template: str,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        enable_thinking: bool = False,
    ) -> str:
        """
        Processes a single instance using vLLM's built-in chat method.
        """
        current_history: List[ChatCompletionMessageParam] = []
        current_outputs: List[str] = []

        if start_template:  # Only add if start_template is non-empty
            current_history.append({"role": "system", "content": start_template})
            logging.debug(
                f"[Instance {instance_idx}] Starting chat history with start template: {start_template[:50]}"
            )
        else:
            logging.debug(
                f"[Instance {instance_idx}] Starting chat history without a start template."
            )

        effective_max_tokens_per_turn = None
        if max_tokens is not None:
            if instance_prompts:  # Ensure there are prompts to divide by
                if enable_thinking:
                    effective_max_tokens_per_turn = max_tokens
                else:
                    effective_max_tokens_per_turn = max_tokens // len(instance_prompts)
            else:
                pass

            if (
                effective_max_tokens_per_turn is not None
                and effective_max_tokens_per_turn <= 0
            ):
                logging.warning(
                    f"[Instance {instance_idx}] Calculated effective_max_tokens per turn is non-positive ({effective_max_tokens_per_turn}) from total max_tokens ({max_tokens}) and {len(instance_prompts)} turns. Setting to default 128 for subsequent turns if any."
                )
                effective_max_tokens_per_turn = 128
        else:
            logging.debug(
                f"[Instance {instance_idx}] max_tokens is None for the entire chat. vLLM will use its default for each turn."
            )

        logging.debug(
            f"[Instance {instance_idx}] Using effective max_tokens per turn: {effective_max_tokens_per_turn}"
        )

        # Process each turn
        for turn_idx, user_prompt in enumerate(instance_prompts):
            user_prompt_processed = user_prompt.strip()

            current_history.append({"role": "user", "content": user_prompt_processed})
            logging.debug(
                f"[Instance {instance_idx}, Turn {turn_idx}] Processing user prompt: '{user_prompt_processed[:50]}'"
            )

            # Create sampling params for this turn
            vllm_sampling_params = VLLM_SamplingParams(
                temperature=temperature,
                top_p=top_p,
                min_tokens=2,
                max_tokens=effective_max_tokens_per_turn
            )

            try:
                request_outputs = None
                # Use vLLM's chat method for single conversation
                if isinstance(self.client, VLLM_Engine):
                    request_outputs = self.client.chat(
                        messages=[
                            current_history
                        ],  # vLLM chat expects list of conversations
                        sampling_params=vllm_sampling_params,
                        use_tqdm=True,  # Disable tqdm as requested
                        chat_template_kwargs={"enable_thinking": enable_thinking},
                    )

                if request_outputs and request_outputs[0].outputs:
                    assistant_response = request_outputs[0].outputs[0].text.strip()
                else:
                    assistant_response = ""

                logging.debug(
                    f"[Instance {instance_idx}, Turn {turn_idx}] Received assistant response: '{assistant_response[:50]}'"
                )

                if not assistant_response:
                    logging.warning(
                        f"[Instance {instance_idx}, Turn {turn_idx}] vLLM response failed or was empty for prompt: '{user_prompt_processed[:100]}'"
                    )
                    current_history.append(
                        {"role": "assistant", "content": "Please Continue..."}
                    )  # last resort to avoid empty history
                    current_outputs.append("NaN")
                else:
                    history_content = self._remove_thinking_traces(assistant_response)
                    current_history.append(
                        {"role": "assistant", "content": history_content}
                    )
                    if history_content == "No Answer":
                        current_outputs.append("NaN")
                    else:
                        current_outputs.append(assistant_response)

            except Exception as e:
                logging.error(
                    f"[Instance {instance_idx}, Turn {turn_idx}] Error during vLLM chat generation: {e}",
                    exc_info=True,
                )
                current_outputs.append("NaN")

        if not current_outputs:
            # This can happen if instance_prompts was empty
            if not instance_prompts and start_template:
                logging.warning(
                    f"[Instance {instance_idx}] No user_prompts provided for this instance, so no chat turns executed beyond potential implicit use of start_template (if any). Returning empty string."
                )
            elif not instance_prompts and not start_template:
                logging.warning(
                    f"[Instance {instance_idx}] No start_template and no user_prompts provided. Returning empty string."
                )
            else:  # instance_prompts was not empty, but all turns failed
                logging.warning(
                    f"[Instance {instance_idx}] No outputs generated for this instance (all turns might have failed)."
                )
            return ""

        return "|".join(current_outputs)

    def generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = 512,
        temperature: float = 0.1,
        top_p: float = 1.0,
        enable_thinking: bool = False
    ) -> List[str]:
        if not prompts:
            return []

        num_prompts = len(prompts)
        logging.info(
            f"Generating {num_prompts} completions with model {self.model_name}..."
        )
        logging.debug(
            f"Sampling params: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}"
        )

        outputs_text: List[str] = [""] * num_prompts
        vllm_sampling_params = VLLM_SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        # Prepare messages for the chat format
        vllm_messages: list[list[ChatCompletionMessageParam]] = [
            [{"role": "user", "content": p}] for p in prompts
        ]

        try:
            # Use the chat method for single-turn generation
            if isinstance(self.client, VLLM_Engine):
                request_outputs = self.client.chat(
                    messages=vllm_messages,
                    sampling_params=vllm_sampling_params,
                    use_tqdm=True,  # Matches behavior in multi-turn
                    chat_template_kwargs={"enable_thinking": enable_thinking},
                )
            
                for i, output in enumerate(request_outputs):
                    if output.outputs:
                        outputs_text[i] = output.outputs[0].text.strip()
                    else:
                        logging.warning(f"vLLM returned no output for prompt index {i}")
                        outputs_text[i] = ""
                return outputs_text
            return [""] * num_prompts
        except Exception as e:
            logging.error(f"Error during vLLM chat generation: {e}", exc_info=True)
            return [""] * num_prompts

    def fill_in_turn(self, instance_history, ground_truth_output, llm_output):
        """
        Fill in the assistant answers for a specific step.
        """

        def replace_answer(content: str, llm_output: str) -> str:
            if content is None:
                raise ValueError("Content cannot be None")
            # replace the answer tags in the llm_output with the ground truth output
            if llm_output is None:
                raise ValueError("LLM output cannot be None")
            logging.debug(
                f"Replacing answer in LLM output: {llm_output} with ground truth output: {ground_truth_output}"
            )
            # split the llm_output by <answer>
            parts = llm_output.split("<answer>")
            logging.debug(
                f"Splitting LLM output into parts: {parts} for ground truth output: {ground_truth_output}"
            )
            # get the first part and add the ground truth output
            llm_out = parts[0] + "<answer>" + str(ground_truth_output) + "</answer>"
            logging.debug(f"Replaced LLM output: {llm_out} ")
            return llm_out

        instance_history.append(
            {
                "role": "assistant",
                "content": replace_answer(ground_truth_output, llm_output),
            }
        )
        return instance_history

    def _implement_sliding_window(
        self,
        history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        if self.sliding_window_size is None or self.sliding_window_size <= 0:
            return history  # infinite window, return full history

        # Handle case where we only have system message or very short history
        if len(history) <= 1:  # Only system message or empty
            return history

        # Count conversation turns (user + assistant pairs) excluding system message
        # Each turn consists of 2 messages: user input + assistant response
        system_msg = history[0] if history[0]["role"] == "system" else None
        conversation_start_idx = 1 if system_msg else 0
        conversation_messages = history[conversation_start_idx:]

        # Calculate how many complete turns we have
        num_complete_turns = len(conversation_messages) // 2

        # If we have fewer turns than the window size, return full history
        if num_complete_turns <= self.sliding_window_size:
            return history

        # Keep only the last sliding_window_size turns (each turn = 2 messages)
        messages_to_keep = self.sliding_window_size * 2
        windowed_conversation = conversation_messages[-messages_to_keep:]

        # Reconstruct history with system message (if exists) + windowed conversation
        if system_msg:
            return [system_msg] + windowed_conversation
        else:
            return windowed_conversation

    def _perform_majority_vote(
        self,
        outputs: List[str],
    ):
        original_outputs = outputs.copy()

        # get the answer from the regex
        def extract_answer(output: str) -> str:
            if output is None:
                return ""
            match = re.search(r"<answer>(.*?)</answer>", output)
            if match:
                return match.group(1).strip()
            else:
                logging.warning(
                    f"Output '{output}' does not contain a valid <answer> tag. Returning empty string."
                )
            return ""

        outputs = [extract_answer(output) for output in outputs]
        # remove invalid outputs
        valid_outputs = []
        for out in outputs:
            try:
                out_int = int(out)
                valid_outputs.append(out_int)
            except ValueError:
                logging.warning(
                    f"Output '{out}' is not a valid integer, removing from majority vote."
                )
        if len(valid_outputs) == 0:
            logging.warning(
                "No valid outputs found for majority vote. returning random answer."
            )
            return f"<answer>{random.choice(original_outputs)}</answer>"

        # perform majority vote with random tie-breaking
        count = Counter(valid_outputs)
        max_count = max(count.values())
        top_answers = [answer for answer, c in count.items() if c == max_count]

        if len(top_answers) == 1:
            chosen = top_answers[0]
            logging.debug(
                f"Majority vote result: {chosen} with count {max_count} (no tie)"
            )
        else:
            chosen = random.choice(top_answers)
            logging.debug(
                f"Majority vote tie among {top_answers} with count {max_count}. Chose {chosen} randomly."
            )

        return f"<answer>{chosen}</answer>"

    def _process_step_wise_vllm_chat(
        self,
        start_template: str,
        user_prompts_list: List[List[str]],
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        enable_thinking: bool = False,
        evaluator: Optional[Any] = None,
        majority_vote: int = 1,
    ) -> List[str]:
        """
        Processes all instances step-by-step using vLLM's built-in chat method.
        For each step, all instances are processed together in a batch.
        """
        if not user_prompts_list:
            return []

        num_instances = len(user_prompts_list)
        max_steps = (
            max(len(prompts) for prompts in user_prompts_list)
            if user_prompts_list
            else 0
        )

        # Initialize conversation histories and outputs for each instance
        instance_histories: List[List[Dict[str, str]]] = [
            [] for _ in range(num_instances)
        ]
        instance_outputs: List[List[str]] = [[] for _ in range(num_instances)]

        # Add start template to all histories if provided
        if start_template:
            for history in instance_histories:
                history.append({"role": "system", "content": start_template})
            logging.debug(
                f"Added start template to all {num_instances} conversation histories"
            )

        # Calculate effective max tokens per turn
        effective_max_tokens_per_turn = None
        if max_tokens is not None:
            if max_steps > 0:
                if enable_thinking:
                    effective_max_tokens_per_turn = max_tokens
                else:
                    effective_max_tokens_per_turn = 10000
            if (
                effective_max_tokens_per_turn is not None
                and effective_max_tokens_per_turn <= 0
            ):
                logging.warning(
                    f"Calculated effective_max_tokens per turn is non-positive ({effective_max_tokens_per_turn}) "
                    f"from total max_tokens ({max_tokens}) and {max_steps} steps. Setting to default 128."
                )
                effective_max_tokens_per_turn = 128

        logging.debug(
            f"Using effective max_tokens per turn: {effective_max_tokens_per_turn}"
        )

        # Process each step across all instances
        for step_idx in tqdm(range(max_steps), desc="Processing steps"):

            num_tokens_generated = 0
            active_messages = []

            for instance_idx, instance_prompts in enumerate(user_prompts_list):
                user_prompt = instance_prompts[step_idx].strip()

                # Add user message to this instance's history
                instance_histories[instance_idx].append(
                    {"role": "user", "content": user_prompt}
                )

                # Add to batch for processing
                active_messages.append(instance_histories[instance_idx].copy())

            logging.debug(f"Step {step_idx}: Processing {num_instances} instances")

            try:
                request_outputs = []
                if majority_vote > 1:
                    for start_vote in range(0, majority_vote, 5):
                        current_n = (
                            5
                            if (majority_vote - start_vote) >= 5
                            else (majority_vote - start_vote)
                        )
                        batch_params = VLLM_SamplingParams(
                            temperature=temperature,
                            top_p=top_p,
                            min_tokens=2,
                            max_tokens=effective_max_tokens_per_turn,
                            n=current_n,
                        )
                        temp_outputs = self.client.chat(  # type: ignore
                            messages=active_messages,
                            sampling_params=batch_params,
                            use_tqdm=False,  # We already have outer progress bar
                            chat_template_kwargs={"enable_thinking": enable_thinking},
                        )
                        request_outputs.extend(temp_outputs)
                else:
                    single_params = VLLM_SamplingParams(
                        temperature=temperature,
                        top_p=top_p,
                        min_tokens=2,
                        max_tokens=effective_max_tokens_per_turn,
                        n=1,
                    )
                    request_outputs = self.client.chat(  # type: ignore
                        messages=active_messages,
                        sampling_params=single_params,
                        use_tqdm=False,  # We already have outer progress bar
                        chat_template_kwargs={"enable_thinking": enable_thinking},
                    )

                # Process results and update histories
                for instance_idx in range(num_instances):
                    if (
                        instance_idx < len(request_outputs)
                        and request_outputs[instance_idx].outputs
                    ):
                        if majority_vote == 1:
                            num_tokens_generated += len(
                                request_outputs[instance_idx].outputs[0].token_ids
                            )
                            assistant_response = (
                                request_outputs[instance_idx].outputs[0].text.strip()
                            )
                        else:
                            # For majority vote, we need to aggregate outputs
                            assistant_responses = [
                                output.text.strip()
                                for output in request_outputs[instance_idx].outputs
                            ]
                            num_tokens_generated += sum(
                                len(output.token_ids)
                                for output in request_outputs[instance_idx].outputs
                            )
                            assistant_response = self._perform_majority_vote(
                                assistant_responses
                            )
                    else:
                        assistant_response = ""

                    if not assistant_response:
                        logging.warning(
                            f"Step {step_idx}, Instance {instance_idx}: vLLM response failed or was empty"
                        )
                        instance_histories[instance_idx].append(
                            {"role": "assistant", "content": "Please Continue..."}
                        )
                        instance_outputs[instance_idx].append("No Answer")
                    else:
                        history_content = self._remove_thinking_traces(
                            assistant_response
                        )

                        instance_histories[instance_idx].append(
                            {"role": "assistant", "content": history_content}
                        )

                        if history_content == "No Answer":
                            logging.warning(
                                f"Step {step_idx}, Instance {instance_idx}: Assistant response was 'No Answer'"
                            )
                            instance_outputs[instance_idx].append("No Answer")
                        else:
                            instance_outputs[instance_idx].append(assistant_response)

                    # Apply sliding window after completing a turn (user + assistant)
                    if self.sliding_window_size is not None:
                        # Count conversation turns: exclude system message, then divide by 2 (user + assistant pairs)
                        system_msg_count = (
                            1
                            if (
                                instance_histories[instance_idx]
                                and instance_histories[instance_idx][0]["role"]
                                == "system"
                            )
                            else 0
                        )
                        conversation_messages = (
                            len(instance_histories[instance_idx]) - system_msg_count
                        )
                        complete_turns = conversation_messages // 2

                        if complete_turns > self.sliding_window_size:
                            instance_histories[instance_idx] = (
                                self._implement_sliding_window(
                                    history=instance_histories[instance_idx],
                                )
                            )

                            logging.debug(
                                f"Applied sliding window to instance {instance_idx} after step {step_idx}. "
                                f"Kept last {self.sliding_window_size} turns, new history length: {len(instance_histories[instance_idx])} messages"
                            )

            except Exception as e:
                logging.error(
                    f"Error during step {step_idx} vLLM chat generation: {e}",
                    exc_info=True,
                )
                # Add error placeholder for all instances
                for instance_idx in range(num_instances):
                    instance_histories[instance_idx].append(
                        {"role": "assistant", "content": "No Answer"}
                    )
                    instance_outputs[instance_idx].append("NaN")

            # evaluate this step
            if evaluator:
                # Use the evaluator's declared step delimiter if available, else default to '|'
                step_delimiter = getattr(
                    getattr(evaluator, "parser", None), "STEP_DELIMITER", "|"
                )
                this_step_outputs = step_delimiter.join(
                    instance_outputs[instance_idx][step_idx]
                    for instance_idx in range(num_instances)
                )

                evaluator.evaluate_step(
                    llm_output=this_step_outputs,
                    step=step_idx,
                    num_tokens_generated=num_tokens_generated,
                    enable_thinking=enable_thinking
                )

                try:
                    acc = evaluator.metrics.step_correctness_array[step_idx]
                    if self.early_stopping and acc <= self.early_stopping_threshold:
                        logging.info(
                            f"Early stopping at step {step_idx} due to step correctness {acc} below threshold {self.early_stopping_threshold}"
                        )
                        break
                except IndexError:
                    logging.warning(
                        f"IndexError accessing prefix correctness for step {step_idx}. "
                        "This might happen if the evaluator has fewer steps than expected."
                    )

            self.tokens_generated_per_step.append(num_tokens_generated)

        # Convert outputs to the expected format ("|" separated strings)
        final_outputs = []
        for instance_idx in range(num_instances):
            if instance_outputs[instance_idx]:
                final_outputs.append("|".join(instance_outputs[instance_idx]))
            else:
                final_outputs.append("")

        return final_outputs

    @timing_decorator
    def chat_generate_step_wise(
        self,
        start_template: str,
        user_prompts_list: List[List[str]],
        evaluator: Optional[Any] = None,
        max_tokens: Optional[int] = 512,
        temperature: float = 0.1,
        top_p: float = 1.0,
        enable_thinking: bool = False,
        majority_vote: int = 1,
    ) -> List[str]:
        """
        Alternative chat generation method that processes step-wise instead of instance-wise.
        For vLLM: processes all instances at each step together for better batching.
        For OpenRouter: falls back to the original method since it's already parallelized.
        """
        if not user_prompts_list:
            logging.warning(
                "Empty user_prompts_list provided to chat_generate_step_wise. Returning empty list."
            )
            return []

        num_instances = len(user_prompts_list)
        logging.info(
            f"Starting step-wise chat generation for {num_instances} instances "
            f"with provider vLLM and model {self.model_name}."
        )

        logging.info(
            "Using step-wise vLLM processing: all instances processed together at each step."
        )
        return self._process_step_wise_vllm_chat(
            start_template=start_template,
            user_prompts_list=user_prompts_list,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
            evaluator=evaluator,
            majority_vote=majority_vote,
        )

    def get_model_name(self) -> str:
        return f"{self.model_name}"
