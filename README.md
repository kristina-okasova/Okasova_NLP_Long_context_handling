# Long-context handling of small LLMs

## Memory experiments
This repository implements an Infini-attention–inspired memory mechanism integrated into the LLaMA architecture. The approach combines local causal attention, responsible for processing the current input segment, with global information retrieved from a persistent memory, enabling the model to incorporate long-term contextual information beyond the standard attention window.

In its original form, the Infini-attention mechanism is known to be numerically unstable. To address this issue, several additional normalization steps are introduced in this implementation. Specifically, queries are normalized during memory retrieval, keys are normalized during memory updates, and both the memory tensor and its associated normalization term are explicitly normalized after each update. These modifications significantly improve training stability and ensure more robust memory accumulation over long sequences.

The method is implemented as a targeted modification of the standard LLaMA model provided by the transformers library. The core architecture, including embeddings, decoder layers, and projection heads, remains unchanged, while the standard self-attention module is replaced with a custom InfiniAttention module. This design preserves full compatibility with pre-trained LLaMA weights while enabling experimentation with memory-augmented attention mechanisms for long-context modeling.

## Illusion of Diminishing Returns
This repository also includes a replication of the evaluation framework proposed in the Illusion of Diminishing Returns line of work, based on the original implementation available at the referenced repository. From the original codebase, only the components strictly necessary for the dictionary-based cumulative summary task are retained, with the evaluation adapted to run efficiently using vLLM for inference.

A key limitation of the original evaluation protocol is addressed in this implementation. In the standard setup, the task repeatedly requires summing retrieved integer values across many steps, which can lead to extremely large or small cumulative totals. This behavior conflates long-context handling with numerical instability, a known weakness of LLMs when operating on large numbers, and can obscure the true source of model errors. To mitigate this issue, the dictionary construction is modified such that only half of the entries are assigned random positive integer values, while the remaining entries contain their exact negated counterparts. All key–value pairs are then randomly mixed. This design keeps the expected magnitude of the running sum bounded and isolates long-context dependency from large-number arithmetic.

Beyond this core setup, several extensions of the evaluation framework are implemented to support future experimentation. These include partial updates of dictionary values, joint updates of both keys and values, periodic or randomly triggered dictionary modifications, complete replacement of the dictionary, and stochastic decisions at each step between executing a summation operation or updating the dictionary. While these variants are implemented and documented, they are not evaluated within the scope of this project and are provided as a foundation for further exploration.
## Fine-tuned models
This repository provides four fine-tuned variants of the LLaMA 3.2 Instruct 1B model augmented with the proposed Infini-attention–based memory mechanism. Each variant explores a different configuration of the memory update and optimization strategy in order to analyze their impact on training stability and long-context performance.

Specifically, the released models cover all combinations of the following design choices:

- Delta-based memory updates enabled or disabled, where the delta variant updates memory using the residual between current value states and values retrieved from memory.

- Gradient flow through memory enabled or disabled, where memory and normalization tensors are either treated as trainable parameters optimized via backpropagation or as non-trainable internal states updated deterministically during the forward pass.

All four variants were fine-tuned under the same experimental setup and evaluated using identical training, validation, and testing protocols. Based on the metrics logged throughout the fine-tuning process, including training loss, validation loss, and perplexity, the best-performing configuration is the model that combines delta-based memory updates with enabled gradient flow through the memory. This variant consistently demonstrated superior optimization behavior and more effective utilization of the memory mechanism during training.

The remaining variants are included for completeness and ablation purposes, enabling further analysis of the individual contributions of delta updates and gradient-based memory optimization.
