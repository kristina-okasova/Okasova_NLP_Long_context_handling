""" Pytorch Llama model"""
import torch
from torch import nn
import torch.nn.functional as F

from transformers import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm, 
    LlamaRotaryEmbedding,
    LlamaPreTrainedModel,
    LlamaDecoderLayer,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.masking_utils import create_causal_mask
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from typing import Optional, Any, cast, Union


class ARMTApproach(nn.Module):
    def __init__(self, num_attention_heads, num_key_value_heads, head_dim, hidden_size, number_of_layers):
        super().__init__()

        self.nu = 3
        self.eps = 1e-8
        self.correction = True

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.dpfp_dim = self.head_dim * 2 * self.nu

        self.W_mb = nn.Linear(self.hidden_size, self.hidden_size)
        self.memory = nn.Parameter(
            torch.zeros(
                number_of_layers,
                self.num_attention_heads,
                self.dpfp_dim,
                self.head_dim
            )
        )

        self.norm = nn.Parameter(
            torch.zeros(
                number_of_layers,
                self.num_attention_heads,
                self.dpfp_dim
            )
        )
        

    def associate(self, hidden_states: torch.Tensor, layer_idx: int, q_proj, memory_gate) -> torch.Tensor:
        memory_layer = self.memory[layer_idx].detach().clone()
        norm_layer = self.norm[layer_idx].detach().clone()
        
        batch_size, seq_len, _ = hidden_states.size()

        query_states = q_proj(hidden_states)
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        memory_hidden_states = self._retrieve_from_memory(
            query_states, 
            memory_layer,
            norm_layer
        )

        memory_hidden_states = memory_hidden_states.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        memory_gate.to(hidden_states.device)
        gate = torch.sigmoid(memory_gate(hidden_states))
        hidden_states = hidden_states + gate * memory_hidden_states

        return hidden_states


    def _retrieve_from_memory(self, Q, memory, norm):
        mq = self.dpfp(Q)
        mq = F.normalize(mq, dim=-1, p=2.0)
        
        num = torch.matmul(mq, memory)
        denom = (mq * norm.unsqueeze(0).unsqueeze(2)).sum(dim=-1, keepdim=True) + self.eps

        return num / denom


    def _update_memory(self, hidden_states: torch.Tensor, layer_idx: int, k_proj, v_proj):
        batch_size, seq_len, _ = hidden_states.size()
        
        key_states = k_proj(hidden_states)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
        mk = self.dpfp(key_states)
        mk = F.normalize(mk, dim=-1, p=2.0)

        value_states = v_proj(hidden_states)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1)

        memory_layer = self.memory[layer_idx].detach().clone()
        norm_layer = self.norm[layer_idx].detach().clone()

        extended_mk = mk.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1)
        num = torch.matmul(extended_mk, memory_layer)

        denom = (extended_mk * norm_layer.unsqueeze(0).unsqueeze(2)).sum(dim=-1, keepdim=True) + self.eps
        previous_value_states = num / denom

        if self.correction:
            new_info_coef = (1 - denom / (torch.linalg.norm(extended_mk, dim=-1) ** 2)[..., None])
            new_info_coef = torch.clip(new_info_coef, 0, 1)
        else:
            new_info_coef = 1
    
        mv = value_states - previous_value_states
        mb = torch.sigmoid(self.W_mb(hidden_states))
        mb = mb.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        weighted_mv = mv * mb
        extended_mk = mk.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1)
        mk_T = extended_mk.transpose(-1, -2)
        associations = torch.matmul(mk_T, weighted_mv)

        delta_memory = associations.squeeze(0)
        delta_norm = (new_info_coef * extended_mk).sum(dim=-2).squeeze(0)

        with torch.no_grad():
            self.memory[layer_idx] = self.memory[layer_idx] + delta_memory
            self.norm[layer_idx] = self.norm[layer_idx] + delta_norm

    def dpfp(self, x):
        x = torch.cat([F.relu(x), F.relu(-x)], dim=-1)
        x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1, self.nu+1)], dim=-1)
        x_repeat = torch.cat([x] * self.nu, dim=-1)
        return x_repeat * x_rolled


class ARMTLlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = torch.nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.armt_approach = ARMTApproach(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            hidden_size=config.hidden_size,
            number_of_layers=config.num_hidden_layers
        )


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds if inputs_embeds is not None else torch.empty(),
            attention_mask=attention_mask,
            cache_position=cache_position if cache_position is not None else torch.empty(),
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if hidden_states is None:
                raise ValueError("hidden_states must not be None")
            
            hidden_states = self.armt_approach.associate(
                hidden_states, 
                cast(int, decoder_layer.self_attn.layer_idx),  # type: ignore
                decoder_layer.self_attn.q_proj,  # type: ignore
                self.armt_memory_gate  # type: ignore
            )
            
            layer_output = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs
            )

            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
            self.armt_approach._update_memory(
                hidden_states, 
                decoder_layer.self_attn.layer_idx,  # type: ignore
                decoder_layer.self_attn.k_proj,  # type: ignore
                decoder_layer.self_attn.v_proj  # type: ignore
            )
    
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        ) 
    

class ARMTLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = ARMTLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if hidden_states is None:
            raise ValueError("hidden_states must not be None")
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None and (labels != -100).sum() != 0:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
            print("Loss contains NaN or Inf")

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def reset_memory(self):
        with torch.no_grad():
            self.model.armt_approach.memory.zero_()  # type: ignore
            self.model.armt_approach.norm.zero_()  # type: ignore

    
    def get_memory(self):
        return self.model.armt_approach.memory
