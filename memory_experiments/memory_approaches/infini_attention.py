""" Pytorch Llama model"""
import torch
from torch import nn
import torch.nn.functional as F
import math

from transformers import DynamicCache, GradientCheckpointingLayer
from transformers.utils.generic import ModelOutput
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm, 
    LlamaRotaryEmbedding, 
    apply_rotary_pos_emb,
    LlamaMLP,
    LlamaPreTrainedModel
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.masking_utils import create_causal_mask
from transformers.generation.utils import GenerationMixin

from dataclasses import dataclass
from typing import Optional, Tuple, Any, Union, cast


@dataclass
class InfiniBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = cast(torch.FloatTensor, torch.empty(0, dtype=torch.float))
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class InfiniCausalLMOutputWithPast(ModelOutput):
    logits: torch.FloatTensor = cast(torch.FloatTensor, torch.empty(0, dtype=torch.float))
    loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    

class CustomLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.rotary_emb = LlamaRotaryEmbedding(config)

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = flash_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            causal=True,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, None


class InfiniAttention(CustomLlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.rotary_emb = LlamaRotaryEmbedding(config)

        self.memory = torch.nn.Parameter(
                torch.zeros(self.config.num_key_value_heads, self.head_dim, self.head_dim)
            )

        self.norm = torch.nn.Parameter(
                torch.zeros(self.config.num_key_value_heads, self.head_dim, self.head_dim)
            )

        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = 1e-8
        self.use_delta = True


    def forward(
            self, 
            hidden_states: torch.Tensor, 
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Any] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        memory_output = self._retrieve_from_memory(
            query_states, 
            self.memory,
            self.norm
        )

        self._update_memory(
            key_states,
            value_states,
            self.memory,
            self.norm
        )

        cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_key_values = getattr(self, "past_key_values", past_key_values)
        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position
            }
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        
        # Compute attention
        attn_flat, _ = flash_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            causal=True,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_flat.reshape(batch_size, self.config.num_attention_heads, seq_len, self.head_dim)
        
        if not torch.isfinite(attn_output).all():
            print("***ERROR***: Attention output NaN")

            print("Q max/min:", query_states.max(), query_states.min())
            print("K max/min:", key_states.max(), key_states.min())
            print("V max/min:", value_states.max(), value_states.min())

        if memory_output is None:
            combined_output = attn_output
        else:
            combined_output = (
                F.sigmoid(self.beta) * memory_output + (1 - F.sigmoid(self.beta)) * attn_output
            )

        combined_output = combined_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.hidden_size)
        final_output = self.o_proj(combined_output)

        return final_output


    def _retrieve_from_memory(self, Q, memory, norm):
        expanded_memory = memory.repeat(1, self.num_key_value_groups, 1, 1)
        
        query_states = F.elu(Q) + 1
        normalized_query_states = query_states / (query_states.sum(dim=-1, keepdim=True) + self.eps)
        retrieved_memory = torch.matmul(normalized_query_states, expanded_memory)

        expanded_norm = norm.transpose(-2, -1).repeat(1, self.num_key_value_groups, 1, 1)
        retrieved_norm = torch.matmul(normalized_query_states, expanded_norm) + self.eps

        memory_output = retrieved_memory / retrieved_norm
        return memory_output


    def _update_memory(self, K, V, memory, norm):
        key_states = F.elu(K) + 1
        # normalize K during update
        normalized_key_states = key_states / (key_states.sum(dim=-1, keepdim=True) + self.eps)
        normalized_key_states_transposed = normalized_key_states.transpose(-2, -1)
        
        if self.use_delta and memory is not None:
            V_retrieved = torch.matmul(normalized_key_states, memory) / (torch.matmul(normalized_key_states, norm.transpose(-2, -1)) + self.eps)
            delta = torch.matmul(normalized_key_states_transposed, V - V_retrieved).sum(dim=0)
        else:
            delta = torch.matmul(normalized_key_states_transposed, V).squeeze(0)
            
        updated_memory = memory + delta if memory is not None else delta

        delta_norm = normalized_key_states.sum(dim=2, keepdim=True).squeeze(0)
        updated_norm = norm + delta_norm if norm is not None else delta

        with torch.no_grad():
            self.memory.copy_(updated_memory)
            self.norm.copy_(updated_norm)
    

class InfiniAttentionDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = InfiniAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Infini-attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # fully connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class InfiniAttentionLlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = torch.nn.ModuleList(
            [InfiniAttentionDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, InfiniBaseModelOutputWithPast]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None and inputs_embeds is not None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = cast(
                torch.LongTensor,
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            )

        if position_ids is None and cache_position is not None:
            position_ids = cast(torch.LongTensor, cache_position.unsqueeze(0))

        hidden_states = inputs_embeds
        _hidden_states = None
        if output_hidden_states:
            _hidden_states = []
        
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states and _hidden_states is not None:
                _hidden_states.append(hidden_states)

            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)
        all_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = (
            tuple(_hidden_states) if output_hidden_states and _hidden_states is not None else None
        )

        return InfiniBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states
        )
    

class InfiniAttentionLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = InfiniAttentionLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs
    ) -> Union[Tuple, InfiniCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position
        )

        hidden_states = outputs[0]

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
            )

        return InfiniCausalLMOutputWithPast(
            logits=logits,
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states
        )
    
    def reset_memory(self):
        with torch.no_grad():
            for layer in self.model.layers:
                for memory in layer.self_attn.memory:  # type: ignore
                    memory.zero_()

                for norm in layer.self_attn.norm:  # type: ignore
                    norm.zero_()


    def get_memory(self):
        return torch.cat([layer.self_attn.memory for layer in self.model.layers], dim=0)  # type: ignore
