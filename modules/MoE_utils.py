from collections import OrderedDict
from typing import Tuple, Union
import os
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from typing import Optional


def load_balancing_loss_func( gate_logits: torch.Tensor, num_experts = 4, top_k=2, attention_mask: Optional[torch.Tensor] = None) -> float:
 
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