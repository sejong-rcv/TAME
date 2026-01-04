# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

import logging
import math
import os
from collections import OrderedDict
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from modules.until_config import PretrainedConfig

logger = logging.getLogger(__name__)



def load_balancing_loss_func( gate_logits: torch.Tensor, num_experts = 4, top_k=2, attention_mask: Optional[torch.Tensor] = None) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    compute_device = gate_logits[0].device
    layer_aux_loss = []
    for layer_gate in gate_logits:
        routing_weights = torch.nn.functional.softmax(layer_gate, dim=-1)

        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

        if attention_mask is None:
            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.mean(routing_weights, dim=0)
        else:
            batch_size, sequence_length = attention_mask.shape
            num_hidden_layers = layer_gate.shape[0] // (batch_size * sequence_length)

            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
            expert_attention_mask = (
                attention_mask[None, :, :, None, None]
                .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
                .reshape(-1, top_k, num_experts)
                .to(compute_device)
            )

            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
                expert_attention_mask, dim=0
            )

            # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
            router_per_expert_attention_mask = (
                attention_mask[None, :, :, None]
                .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
                .reshape(-1, num_experts)
                .to(compute_device)
            )

            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
                router_per_expert_attention_mask, dim=0
            )

        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
        layer_aux_loss.append(overall_loss)
    
    overall_loss = sum(layer_aux_loss)
    return overall_loss * num_experts

def _split_in_proj(t: torch.Tensor):
    D = t.shape[-1] if t.dim() == 2 else t.numel() // 3
    if t.dim() == 2:
        q, k, v = t[:D, :], t[D:2*D, :], t[2*D:, :]
    else:
        q, k, v = t[:D], t[D:2*D], t[2*D:]
    return q, k, v


def remap_state_for_clipattention(
    model,
    state_dict: dict,
    prefix: str = "clip.visual.transformer.resblocks",
    overwrite_existing_qkv: bool = True,
    remove_inproj_after_split: bool = True,
):
    """
    Convert MultiheadAttention-style parameters (in_proj_*) to CLIPAttention format (q/k/v).

    This function scans transformer blocks whose attention layer has q_proj/k_proj/v_proj
    and, if needed, splits in_proj_weight / in_proj_bias into separate q/k/v parameters.

    Args:
        model: Model instance with model.clip.visual.transformer.resblocks.
        state_dict: State dict to be converted (updated in-place and returned).
        prefix: Base prefix for resblocks in the state dict.
        overwrite_existing_qkv: If True, overwrite q/k/v even if already present.
        remove_inproj_after_split: If True, remove in_proj_* keys after splitting.

    Returns:
        Updated state dict with q/k/v parameters filled in.
    """
    resblocks = model.clip.visual.transformer.resblocks

    for i, block in enumerate(resblocks):
        attn = getattr(block, "attn", None)
        if attn is None:
            continue

        # Only convert when CLIPAttention-style projections are present
        has_qkv = all(hasattr(attn, name) for name in ["q_proj", "k_proj", "v_proj"])
        if not has_qkv:
            continue

        base = f"{prefix}.{i}.attn."

        in_w_key = base + "in_proj_weight"
        in_b_key = base + "in_proj_bias"

        q_w_key = base + "q_proj.weight"
        k_w_key = base + "k_proj.weight"
        v_w_key = base + "v_proj.weight"

        q_b_key = base + "q_proj.bias"
        k_b_key = base + "k_proj.bias"
        v_b_key = base + "v_proj.bias"

        # Convert weights
        if in_w_key in state_dict:
            q_w, k_w, v_w = _split_in_proj(state_dict[in_w_key])

            if overwrite_existing_qkv or not all(k in state_dict for k in [q_w_key, k_w_key, v_w_key]):
                state_dict[q_w_key] = q_w.clone()
                state_dict[k_w_key] = k_w.clone()
                state_dict[v_w_key] = v_w.clone()

            if remove_inproj_after_split:
                del state_dict[in_w_key]

        # Convert biases
        if in_b_key in state_dict:
            q_b, k_b, v_b = _split_in_proj(state_dict[in_b_key])

            if overwrite_existing_qkv or not all(k in state_dict for k in [q_b_key, k_b_key, v_b_key]):
                state_dict[q_b_key] = q_b.clone()
                state_dict[k_b_key] = k_b.clone()
                state_dict[v_b_key] = v_b.clone()

            if remove_inproj_after_split:
                del state_dict[in_b_key]

    return state_dict

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

def _split_in_proj(t: torch.Tensor):
    # [3*D, D] or [3*D] -> (q,k,v)
    if t.dim() == 2:
        D = t.shape[1]
        return t[:D, :], t[D:2*D, :], t[2*D:, :]
    else:
        D = t.numel() // 3
        return t[:D], t[D:2*D], t[2*D:]

def load_clipattention_qkv_from_state(attn, state_dict, base_key: str):
    qw, qb = base_key + "attn.q_proj.weight", base_key + "attn.q_proj.bias"
    kw, kb = base_key + "attn.k_proj.weight", base_key + "attn.k_proj.bias"
    vw, vb = base_key + "attn.v_proj.weight", base_key + "attn.v_proj.bias"
    iw, ib = base_key + "in_proj_weight", base_key + "in_proj_bias"
    ow, ob = base_key + "attn.out_proj.weight", base_key + "attn.out_proj.bias"

    with torch.no_grad():
        if ow in state_dict:
            attn.out_proj.weight.copy_(state_dict[ow].to(attn.out_proj.weight.dtype))
        if ob in state_dict and attn.out_proj.bias is not None:
            attn.out_proj.bias.copy_(state_dict[ob].to(attn.out_proj.bias.dtype))

        has_qkv_w = all(k in state_dict for k in [qw, kw, vw])
        if has_qkv_w:
            attn.q_proj.weight.copy_(state_dict[qw].to(attn.q_proj.weight.dtype))
            attn.k_proj.weight.copy_(state_dict[kw].to(attn.k_proj.weight.dtype))
            attn.v_proj.weight.copy_(state_dict[vw].to(attn.v_proj.weight.dtype))
            if qb in state_dict and attn.q_proj.bias is not None:
                attn.q_proj.bias.copy_(state_dict[qb].to(attn.q_proj.bias.dtype))
            if kb in state_dict and attn.k_proj.bias is not None:
                attn.k_proj.bias.copy_(state_dict[kb].to(attn.k_proj.bias.dtype))
            if vb in state_dict and attn.v_proj.bias is not None:
                attn.v_proj.bias.copy_(state_dict[vb].to(attn.v_proj.bias.dtype))
            return

        if iw in state_dict:
            q_w, k_w, v_w = _split_in_proj(state_dict[iw])
            attn.q_proj.weight.copy_(q_w.to(attn.q_proj.weight.dtype))
            attn.k_proj.weight.copy_(k_w.to(attn.k_proj.weight.dtype))
            attn.v_proj.weight.copy_(v_w.to(attn.v_proj.weight.dtype))

        if ib in state_dict:
            q_b, k_b, v_b = _split_in_proj(state_dict[ib])
            if attn.q_proj.bias is not None:
                attn.q_proj.bias.copy_(q_b.to(attn.q_proj.bias.dtype))
            if attn.k_proj.bias is not None:
                attn.k_proj.bias.copy_(k_b.to(attn.k_proj.bias.dtype))
            if attn.v_proj.bias is not None:
                attn.v_proj.bias.copy_(v_b.to(attn.v_proj.bias.dtype))

class PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def resize_token_embeddings(self, new_num_tokens=None):
        raise NotImplementedError

    @classmethod
    def init_preweight(cls, model, state_dict, prefix=None, task_config=None):
        
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        
        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)


        #  MoE 가중치 별도 초기화 (use_moe 여부에 따라 다르게)
        
        if hasattr(task_config, "use_moe") and getattr(task_config, "use_moe", False):
            # Visual MoE
            if hasattr(model, "clip") and hasattr(model.clip, "visual") and hasattr(model.clip.visual, "transformer"):
                resblocks = model.clip.visual.transformer.resblocks
                moe_layers = getattr(task_config, "visual_MoE_args", [0])[-1]
                total_layers = getattr(task_config, "visual_num_hidden_layers", len(resblocks))
                if moe_layers > 0:
                    normal_idx = total_layers - moe_layers
                    for i in range(normal_idx, total_layers):
                        moe_block = resblocks[i]
                        block_prefix = f"clip.visual.transformer.resblocks.{i}."

                        moe_block.attn.in_proj_weight.data.copy_(state_dict[block_prefix + "attn.in_proj_weight"])
                        moe_block.attn.in_proj_bias.data.copy_(state_dict[block_prefix + "attn.in_proj_bias"])
                        moe_block.attn.out_proj.weight.data.copy_(state_dict[block_prefix + "attn.out_proj.weight"])
                        moe_block.attn.out_proj.bias.data.copy_(state_dict[block_prefix + "attn.out_proj.bias"])
                        moe_block.ln_1.weight.data.copy_(state_dict[block_prefix + "ln_1.weight"])
                        moe_block.ln_1.bias.data.copy_(state_dict[block_prefix + "ln_1.bias"])
                        moe_block.ln_2.weight.data.copy_(state_dict[block_prefix + "ln_2.weight"])
                        moe_block.ln_2.bias.data.copy_(state_dict[block_prefix + "ln_2.bias"])

                        for expert in moe_block.experts:
                            expert[0].weight.data.copy_(state_dict[block_prefix + "mlp.c_fc.weight"])
                            expert[0].bias.data.copy_(state_dict[block_prefix + "mlp.c_fc.bias"])
                            expert[-1].weight.data.copy_(state_dict[block_prefix + "mlp.c_proj.weight"])
                            expert[-1].bias.data.copy_(state_dict[block_prefix + "mlp.c_proj.bias"])

     

                    logger.info(f"Visual MoE experts initialized from last normal MLP at idx {normal_idx}")

            # Text MoE
            if hasattr(model, "clip") and hasattr(model.clip, "transformer"):
                resblocks = model.clip.transformer.resblocks
                moe_layers = getattr(task_config, "text_MoE_args", [0])[-1]
                total_layers = getattr(task_config, "text_num_hidden_layers", len(resblocks))
                if moe_layers > 0:
                    normal_idx = total_layers - moe_layers
                    for i in range(normal_idx, total_layers):
                        moe_block = resblocks[i]
                        block_prefix = f"clip.transformer.resblocks.{i}."

                        moe_block.attn.in_proj_weight.data.copy_(state_dict[block_prefix + "attn.in_proj_weight"])
                        moe_block.attn.in_proj_bias.data.copy_(state_dict[block_prefix + "attn.in_proj_bias"])
                        moe_block.attn.out_proj.weight.data.copy_(state_dict[block_prefix + "attn.out_proj.weight"])
                        moe_block.attn.out_proj.bias.data.copy_(state_dict[block_prefix + "attn.out_proj.bias"])
                        moe_block.ln_1.weight.data.copy_(state_dict[block_prefix + "ln_1.weight"])
                        moe_block.ln_1.bias.data.copy_(state_dict[block_prefix + "ln_1.bias"])
                        moe_block.ln_2.weight.data.copy_(state_dict[block_prefix + "ln_2.weight"])
                        moe_block.ln_2.bias.data.copy_(state_dict[block_prefix + "ln_2.bias"])

                        for expert in moe_block.experts:
                            expert[0].weight.data.copy_(state_dict[block_prefix + "mlp.c_fc.weight"])
                            expert[0].bias.data.copy_(state_dict[block_prefix + "mlp.c_fc.bias"])
                            expert[-1].weight.data.copy_(state_dict[block_prefix + "mlp.c_proj.weight"])
                            expert[-1].bias.data.copy_(state_dict[block_prefix + "mlp.c_proj.bias"])
                    logger.info(f"Text MoE experts initialized from last normal MLP at idx {normal_idx}")


        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='')
       
        if prefix is None and (task_config is None or task_config.local_rank == 0):
            logger.info("-" * 20)
            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
            if len(error_msgs) > 0:
                logger.error("Weights from pretrained model cause errors in {}: {}"
                             .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

        return model
    
    @classmethod
    def initialize_moe_weights(model, moe_layers):
        """
        MoEResidualAttentionBlock의 experts와 gate를 초기화합니다.
        
        Args:
            model: CLIP 모델
            moe_layers: MoEResidualAttentionBlock이 있는 레이어 인덱스 리스트
        """
        for layer_idx in moe_layers:
            moe_block = model.transformer.resblocks[layer_idx]
            for expert in moe_block.experts:
                nn.init.xavier_uniform_(expert.c_fc.weight)
                nn.init.xavier_uniform_(expert.c_proj.weight)
                nn.init.zeros_(expert.c_fc.bias)
                nn.init.zeros_(expert.c_proj.bias)
            nn.init.xavier_uniform_(moe_block.gate.weight)

    @classmethod
    def load_weights_for_moe(cls, model, state_dict,prefix=None, task_config=None):
        
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
 
        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)
        
        if hasattr(task_config, "use_moe") and getattr(task_config, "use_moe", False):
            # Visual MoE
            if hasattr(model, "clip") and hasattr(model.clip, "visual") and hasattr(model.clip.visual, "transformer"):
                resblocks = model.clip.visual.transformer.resblocks
                moe_layers = task_config.visual_moe_indices
                if len(moe_layers) > 0:
                    for i in moe_layers:
                        moe_block = resblocks[i]
                        block_prefix = f"clip.visual.transformer.resblocks.{i}."

                        moe_block.attn.in_proj_weight.data.copy_(state_dict[block_prefix + "attn.in_proj_weight"])
                        moe_block.attn.in_proj_bias.data.copy_(state_dict[block_prefix + "attn.in_proj_bias"])
                        moe_block.attn.out_proj.weight.data.copy_(state_dict[block_prefix + "attn.out_proj.weight"])
                        moe_block.attn.out_proj.bias.data.copy_(state_dict[block_prefix + "attn.out_proj.bias"])
                        moe_block.ln_1.weight.data.copy_(state_dict[block_prefix + "ln_1.weight"])
                        moe_block.ln_1.bias.data.copy_(state_dict[block_prefix + "ln_1.bias"])
                        moe_block.ln_2.weight.data.copy_(state_dict[block_prefix + "ln_2.weight"])
                        moe_block.ln_2.bias.data.copy_(state_dict[block_prefix + "ln_2.bias"])

                        for expert in moe_block.experts:
                            expert[0].weight.data.copy_(state_dict[block_prefix + "mlp.c_fc.weight"])
                            expert[0].bias.data.copy_(state_dict[block_prefix + "mlp.c_fc.bias"])
                            expert[-1].weight.data.copy_(state_dict[block_prefix + "mlp.c_proj.weight"])
                            expert[-1].bias.data.copy_(state_dict[block_prefix + "mlp.c_proj.bias"])
     
                    logger.info(f"Visual MoE experts initialized from last normal MLP at idx {moe_layers}")

            # Text MoE
            if hasattr(model, "clip") and hasattr(model.clip, "transformer"):
                resblocks = model.clip.transformer.resblocks
                # moe_layers = getattr(task_config, "text_MoE_args", [0])[-1]
                # total_layers = getattr(task_config, "text_num_hidden_layers", len(resblocks))
                moe_layers = task_config.text_moe_indices
                if len(moe_layers) > 0:
                    
                    for i in moe_layers:
                        moe_block = resblocks[i]
                        block_prefix = f"clip.transformer.resblocks.{i}."

                        moe_block.attn.in_proj_weight.data.copy_(state_dict[block_prefix + "attn.in_proj_weight"])
                        moe_block.attn.in_proj_bias.data.copy_(state_dict[block_prefix + "attn.in_proj_bias"])
                        moe_block.attn.out_proj.weight.data.copy_(state_dict[block_prefix + "attn.out_proj.weight"])
                        moe_block.attn.out_proj.bias.data.copy_(state_dict[block_prefix + "attn.out_proj.bias"])
                        moe_block.ln_1.weight.data.copy_(state_dict[block_prefix + "ln_1.weight"])
                        moe_block.ln_1.bias.data.copy_(state_dict[block_prefix + "ln_1.bias"])
                        moe_block.ln_2.weight.data.copy_(state_dict[block_prefix + "ln_2.weight"])
                        moe_block.ln_2.bias.data.copy_(state_dict[block_prefix + "ln_2.bias"])

                        for expert in moe_block.experts:
                            expert[0].weight.data.copy_(state_dict[block_prefix + "mlp.c_fc.weight"])
                            expert[0].bias.data.copy_(state_dict[block_prefix + "mlp.c_fc.bias"])
                            expert[-1].weight.data.copy_(state_dict[block_prefix + "mlp.c_proj.weight"])
                            expert[-1].bias.data.copy_(state_dict[block_prefix + "mlp.c_proj.bias"])
                    logger.info(f"Text MoE experts initialized from last normal MLP at idx {moe_layers}")


        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='')
       
        if prefix is None and (task_config is None or task_config.local_rank == 0):
            logger.info("-" * 20)
            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
            if len(error_msgs) > 0:
                logger.error("Weights from pretrained model cause errors in {}: {}"
                             .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))
        
        return model
        
    @classmethod
    def load_weights_for_moe_vip(cls, model, state_dict,prefix=None, task_config=None):
        
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
 
        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        
        if hasattr(task_config, "use_moe") and getattr(task_config, "use_moe", False):
            
            # Visual MoE
            if hasattr(model, "clip") and hasattr(model.clip, "visual") and hasattr(model.clip.visual, "transformer"):
                resblocks = model.clip.visual.transformer.resblocks
                moe_layers = task_config.visual_moe_indices
                if len(moe_layers) > 0:
                    for i in moe_layers:

                        moe_block = resblocks[i]
                        block_prefix = f"clip.visual.transformer.resblocks.{i}."

                        load_clipattention_qkv_from_state(moe_block.attn, state_dict, block_prefix)
                       
                        moe_block.attn.out_proj.weight.data.copy_(state_dict[block_prefix + "attn.out_proj.weight"])
                        moe_block.attn.out_proj.bias.data.copy_(state_dict[block_prefix + "attn.out_proj.bias"])
                        moe_block.ln_1.weight.data.copy_(state_dict[block_prefix + "ln_1.weight"])
                        moe_block.ln_1.bias.data.copy_(state_dict[block_prefix + "ln_1.bias"])
                        moe_block.ln_2.weight.data.copy_(state_dict[block_prefix + "ln_2.weight"])
                        moe_block.ln_2.bias.data.copy_(state_dict[block_prefix + "ln_2.bias"])

                        for expert in moe_block.experts:
                            expert[0].weight.data.copy_(state_dict[block_prefix + "mlp.c_fc.weight"])
                            expert[0].bias.data.copy_(state_dict[block_prefix + "mlp.c_fc.bias"])
                            expert[-1].weight.data.copy_(state_dict[block_prefix + "mlp.c_proj.weight"])
                            expert[-1].bias.data.copy_(state_dict[block_prefix + "mlp.c_proj.bias"])
     
                    logger.info(f"Visual MoE experts initialized from last normal MLP at idx {moe_layers}")

            # Text MoE
            if hasattr(model, "clip") and hasattr(model.clip, "transformer"):
                resblocks = model.clip.transformer.resblocks
                # moe_layers = getattr(task_config, "text_MoE_args", [0])[-1]
                # total_layers = getattr(task_config, "text_num_hidden_layers", len(resblocks))
                moe_layers = task_config.text_moe_indices
                if len(moe_layers) > 0:
                    for i in moe_layers:
                        moe_block = resblocks[i]
                        block_prefix = f"clip.transformer.resblocks.{i}."
                        
                        load_clipattention_qkv_from_state(moe_block.attn, state_dict, block_prefix)
                        
                        moe_block.attn.out_proj.weight.data.copy_(state_dict[block_prefix + "attn.out_proj.weight"])
                        moe_block.attn.out_proj.bias.data.copy_(state_dict[block_prefix + "attn.out_proj.bias"])
                        moe_block.ln_1.weight.data.copy_(state_dict[block_prefix + "ln_1.weight"])
                        moe_block.ln_1.bias.data.copy_(state_dict[block_prefix + "ln_1.bias"])
                        moe_block.ln_2.weight.data.copy_(state_dict[block_prefix + "ln_2.weight"])
                        moe_block.ln_2.bias.data.copy_(state_dict[block_prefix + "ln_2.bias"])

                        for expert in moe_block.experts:
                            expert[0].weight.data.copy_(state_dict[block_prefix + "mlp.c_fc.weight"])
                            expert[0].bias.data.copy_(state_dict[block_prefix + "mlp.c_fc.bias"])
                            expert[-1].weight.data.copy_(state_dict[block_prefix + "mlp.c_proj.weight"])
                            expert[-1].bias.data.copy_(state_dict[block_prefix + "mlp.c_proj.bias"])
                    logger.info(f"Text MoE experts initialized from last normal MLP at idx {moe_layers}")


        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        
        load(model, prefix='')
       
        if prefix is None and (task_config is None or task_config.local_rank == 0):
            logger.info("-" * 20)
            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
            if len(error_msgs) > 0:
                logger.error("Weights from pretrained model cause errors in {}: {}"
                             .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))
        
        return model

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    @classmethod
    def from_pretrained(cls, config, state_dict=None,  *inputs, **kwargs):
        """
        Instantiate a PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            return model
        
        model = cls.init_preweight(model, state_dict)

        return model

##################################
###### LOSS FUNCTION #############
##################################
class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

class MILNCELoss(nn.Module):
    def __init__(self, batch_size=1, n_pair=1,):
        super(MILNCELoss, self).__init__()
        self.batch_size = batch_size
        self.n_pair = n_pair
        torch_v = float(".".join(torch.__version__.split(".")[:2]))
        self.bool_dtype = torch.bool if torch_v >= 1.3 else torch.uint8

    def forward(self, sim_matrix):
        mm_mask = np.eye(self.batch_size)
        mm_mask = np.kron(mm_mask, np.ones((self.n_pair, self.n_pair)))
        mm_mask = torch.tensor(mm_mask).float().to(sim_matrix.device)

        from_text_matrix = sim_matrix + mm_mask * -1e12
        from_video_matrix = sim_matrix.transpose(1, 0)

        new_sim_matrix = torch.cat([from_video_matrix, from_text_matrix], dim=-1)
        logpt = F.log_softmax(new_sim_matrix, dim=-1)

        mm_mask_logpt = torch.cat([mm_mask, torch.zeros_like(mm_mask)], dim=-1)
        masked_logpt = logpt + (torch.ones_like(mm_mask_logpt) - mm_mask_logpt) * -1e12

        new_logpt = -torch.logsumexp(masked_logpt, dim=-1)

        logpt_choice = torch.zeros_like(new_logpt)
        mark_ind = torch.arange(self.batch_size).to(sim_matrix.device) * self.n_pair + (self.n_pair//2)
        logpt_choice[mark_ind] = 1
        sim_loss = new_logpt.masked_select(logpt_choice.to(dtype=self.bool_dtype)).mean()
        return sim_loss

class MaxMarginRankingLoss(nn.Module):
    def __init__(self,
                 margin=0.2,
                 negative_weighting=True,
                 batch_size=128,
                 n_pair=1,
                 hard_negative_rate=0.5,
        ):
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin
        self.n_pair = n_pair
        self.batch_size = batch_size
        easy_negative_rate = 1 - hard_negative_rate
        self.easy_negative_rate = easy_negative_rate
        self.negative_weighting = negative_weighting
        if n_pair > 1 and batch_size > 1:
            alpha = easy_negative_rate / ((batch_size - 1) * (1 - easy_negative_rate))
            mm_mask = (1 - alpha) * np.eye(self.batch_size) + alpha
            mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))
            mm_mask = torch.tensor(mm_mask) * (batch_size * (1 - easy_negative_rate))
            self.mm_mask = mm_mask.float()

    def forward(self, x):
        d = torch.diag(x)
        max_margin = F.relu(self.margin + x - d.view(-1, 1)) + \
                     F.relu(self.margin + x - d.view(1, -1))
        if self.negative_weighting and self.n_pair > 1 and self.batch_size > 1:
            max_margin = max_margin * self.mm_mask.to(max_margin.device)
        return max_margin.mean()

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )
