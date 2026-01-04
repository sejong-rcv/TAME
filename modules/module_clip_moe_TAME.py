"""
Adapted from: https://github.com/openai/CLIP/blob/main/clip/clip.py
"""
from collections import OrderedDict
from typing import Tuple, Union

import hashlib
import os
import urllib
import warnings
from tqdm import tqdm
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
import pdb
from typing import Any, Optional, Tuple, Union

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}
_PT_NAME = {
    "RN50": "RN50.pt",
    "RN101": "RN101.pt",
    "RN50x4": "RN50x4.pt",
    "RN50x16": "RN50x16.pt",
    "ViT-B/32": "ViT-B-32.pt",
    "ViT-B/16": "ViT-B-16.pt",
    "ViT-L/14": "ViT-L-14.pt",
    "ViT-L/14-336": "ViT-L-14-336px.pt"
}


def download_model(pretrained_clip_name):
    root = 'modules'
    # root = '/home/hpluo/lhp/Codes/PycharmCodes/Video-Text-Retrieval/CLIP-models'
    model_path = os.path.join(root, _PT_NAME[pretrained_clip_name])
    return model_path

def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):

    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url).read() as source, open(download_target, "wb") as output:   # add .read()
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def available_models():
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None,args=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)      
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))   # LND

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x_tuple:tuple,i):
        
        x, video_frame = x_tuple
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return (x, video_frame)

class V_ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None, args=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc",   nn.Linear(d_model, d_model * 4)),
            ("gelu",   QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.add_cls_num = args.add_cls_num  

    def _run_mha(self, x: torch.Tensor, attn_mask_: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attn_mask_ is not None:
            attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    @torch.no_grad()
    def _check_B(self, N: int, F: int) -> int:
        assert F > 0 and (N % F == 0), f"N={N} must be divisible by frames={F}"
        return N // F
    
    def forward(self, x_tuple: tuple, i):
        x, video_frame = x_tuple
        L, N, E = x.shape
        
        x_norm = self.ln_1(x)  # (L, N, E)

        attn_mask_ = None
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(L)
        attn_in = self._run_mha(x_norm, attn_mask_)  # (L, N, E)
        x = x + attn_in

        x = x + self.mlp(self.ln_2(x))
        return (x, video_frame)

class V_MoEResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None ,args=None):
        super().__init__()

        self.num_experts = args.num_experts        
        self.top_k = args.vis_top_k                    
        self.dropout = args.moe_dropout            

        self.attn = nn.MultiheadAttention(d_model, n_head)    

        self.ln_1 = LayerNorm(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        # MLP Experts
        self.experts = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("dropout", nn.Dropout(p=self.dropout)),
                ("c_proj", nn.Linear(d_model * 4, d_model))
            ])) for _ in range(self.num_experts)
        ])
        
        # Gating network for expert routing
        self.gate = nn.Linear(d_model, self.num_experts, bias=False)
        #add_cls_token_num
        self.add_cls_num = args.add_cls_num
        self.not_use_global_over_frames = getattr(args, "not_use_global_over_frames", False)

    def _run_mha(self, x: torch.Tensor, attn_mask_: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (L, N, E)
        if attn_mask_ is not None:
            attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device)
        out = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]  # (L, N, E)
        return out

    @torch.no_grad()
    def _check_B(self, N: int, F: int) -> int:
        assert F > 0 and (N % F == 0), f"N={N} must be divisible by frames={F}"
        return N // F

    def _global_over_frames_extras_only(self, x_norm: torch.Tensor, num_cls: int, frames: int) -> torch.Tensor:
        L, N, E = x_norm.shape
        B = self._check_B(N, frames)
        C = num_cls
        assert 1 <= C <= L, f"num_cls={C} must be in [1, L={L}]"
        if C - 1 == 0:
            return torch.zeros_like(x_norm)

        t = x_norm[1:, :, :]  # (L-1, N, E)
        t = t.reshape(L-1, B, frames, E).permute(0, 2, 1, 3).reshape((L-1)*frames, B, E).contiguous()
        t = self._run_mha(t)  # ((L-1)*F, B, E)
        t = t.reshape(L-1, frames, B, E).permute(0, 2, 1, 3).reshape(L-1, N, E).contiguous()

        delta_extras = torch.zeros_like(x_norm)  # (L, N, E)
        delta_extras[1:C, :, :] = t[:(C-1), :, :]  
        return delta_extras

    def forward(self, x_tuple: tuple, i):

        # x_tuple: (x, video_frame)
        x, video_frame = x_tuple
        L, N, E = x.shape

        x_norm = self.ln_1(x)                           # (L, N, E)

        attn_mask_ = None
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(L)              
        attn_in = self._run_mha(x_norm, attn_mask_)     # (L, N, E)

        num_cls_total = 1 + self.add_cls_num

        if (not self.not_use_global_over_frames) and (num_cls_total > 1):
            attn_g_extras = self._global_over_frames_extras_only(x_norm, num_cls=num_cls_total, frames=video_frame)
            x = x + attn_in + attn_g_extras
        else:
            x = x + attn_in
        
        hidden_states = self.ln_2(x)

        seq_len, batch_size, hidden_dim = hidden_states.shape 
        
        if self.gate.weight.dtype != hidden_states.dtype :
            hidden_states = hidden_states.to(dtype=torch.float16)

        router_logits = self.gate(hidden_states[0])  # (batch*seq_len, num_experts)
        
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
    
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    
        frame_hidden_states = hidden_states.permute(1,0,2) #-> 48,50,768

        final_hidden_states = torch.zeros(
            (batch_size, seq_len, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])  # idx: which top_k, top_x: which sample 
            if len(top_x) == 0:
                continue
            
            current_state = frame_hidden_states[top_x]
            weight = routing_weights[top_x, idx].view(-1, 1, 1)  # [num_selected, 1, 1]
            current_hidden_states = expert_layer(current_state) * weight 
            
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        final_hidden_states = final_hidden_states.permute(1,0,2) 
        x = x + final_hidden_states
        return (x,video_frame, router_logits)
    
class MoEResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None ,args=None):
        super().__init__()

        self.num_experts = args.num_experts        
        self.top_k = args.top_k                    
        self.dropout = args.moe_dropout            

        self.attn = nn.MultiheadAttention(d_model, n_head)      
        self.ln_1 = LayerNorm(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        # MLP Experts
        self.experts = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("dropout", nn.Dropout(p=self.dropout)),
                ("c_proj", nn.Linear(d_model * 4, d_model))
            ])) for _ in range(self.num_experts)
        ])
        
        # Gating network for expert routing
        self.gate = nn.Linear(d_model, self.num_experts, bias=False)

    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))   # LND

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x_tuple: tuple,i):
        # x_tuple: (x, video_frame)
    
        x, video_frame = x_tuple
        x = x + self.attention(self.ln_1(x))
        hidden_states = self.ln_2(x)
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # MoE Routing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)  # (batch*seq_len, hidden_dim)
        router_logits = self.gate(hidden_states_flat)  # (batch*seq_len, num_experts)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)  # (batch*seq_len, top_k)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states_flat.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim), dtype=hidden_states_flat.dtype, device=hidden_states_flat.device
        )

        # One-hot mask for experts
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # expert_mask: (num_experts, top_k, batch*seq_len)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])  # idx: which top_k, top_x: which sample
            if len(top_x) == 0:
                continue
            current_state = hidden_states_flat[top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states_flat.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        x = x + final_hidden_states
        return (x,video_frame, router_logits)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None, args = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask,args=args) for _ in range(layers)])

    def forward(self, x: torch.Tensor, video_frame=-1):
        return self.resblocks((x, video_frame))[0]

class MoETransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask=None, args=None, encoder_type="visual"):
        super().__init__()
        self.width = width
        self.layers = layers
        self.encoder_type = encoder_type    
        self.use_moe = args.use_moe
        if encoder_type == "visual":
            self.moe_indices = args.visual_moe_indices
        elif encoder_type == "text":
            self.moe_indices = args.text_moe_indices
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        if self.moe_indices is None:
            self.moe_indices = []
        elif isinstance(self.moe_indices, str):
            self.moe_indices = ast.literal_eval(self.moe_indices)

        blocks = []
        
        if encoder_type == "visual":
            for i in range(layers):
                if i in self.moe_indices:
                    blocks.append(V_MoEResidualAttentionBlock(width, heads, attn_mask, args=args))
                else:
                    blocks.append(V_ResidualAttentionBlock(width, heads, attn_mask,args=args))
            self.resblocks = nn.ModuleList(blocks)
        elif encoder_type == "text":
            for i in range(layers):
                if i in self.moe_indices:
                    blocks.append(MoEResidualAttentionBlock(width, heads, attn_mask, args=args))
                else:
                    blocks.append(ResidualAttentionBlock(width, heads, attn_mask,args=args))
            self.resblocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, video_frame=-1):
        router_logits = []

        if not isinstance(x, tuple):
            x_tuple = (x, video_frame)
        else:
            x_tuple = x
        for i, block in enumerate(self.resblocks):
            if i in self.moe_indices:
                x_tuple = block(x_tuple,i)
                x_temp, video_frame_temp, router_logit = x_tuple
                router_logits.append(router_logit)

                x_tuple = (x_temp, video_frame_temp)
            else:
                x_tuple = block(x_tuple,i)


        x, video_frame = x_tuple
        if router_logits:
            return x, video_frame, torch.stack(router_logits)
        else:
            return x, video_frame, torch.empty(0)

class VisualTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        linear_patch: str = '2d',
        args = None
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        
        self.add_cls_num = int(getattr(args, "add_cls_num", 0))
        if self.add_cls_num > 0:
            self.added_cls = nn.Parameter(scale * torch.randn(self.add_cls_num, width))  # [add_cls_num, width]
        else:
            self.added_cls = None  
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.video_frames = args.video_frames
        self.use_moe = args.use_moe

        if self.use_moe and len(args.visual_moe_indices) > 0:
            self.transformer = MoETransformer(width, layers, heads, attn_mask=None, args=args, encoder_type = 'visual')
        else:
            self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # For 3D
        assert linear_patch in ['2d', '3d']
        self.linear_patch = linear_patch
        if self.linear_patch == '3d':
            self.conv2 = nn.Conv3d(
                in_channels=3,
                out_channels=width,
                kernel_size=(3, patch_size, patch_size),
                stride=(1, patch_size, patch_size),
                padding=(1, 0, 0),
                bias=False,
            )

    def forward(self, x: torch.Tensor, video_frame=-1):
        if self.linear_patch == '3d':
            assert video_frame != -1
            x_3d = x.reshape(-1, video_frame, x.shape[-3], x.shape[-2], x.shape[-1])
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            x_3d = self.conv2(x_3d)     # [B, width, F, G, G]
            x_3d = x_3d.permute(0, 2, 1, 3, 4)  # [B, F, width, G, G]
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2], x_3d.shape[-1]).contiguous()
        else:
            x = self.conv1(x)  # [B*F, width, G, G]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B*F, width, grid**2]
        x = x.permute(0, 2, 1)                     # [B*F, grid**2, width]

        batch_size = x.shape[0]
        width = x.shape[-1]

        # [CLS]
        cls_tokens = self.class_embedding.to(x.dtype).unsqueeze(0).expand(batch_size, 1, -1)  # [B,1,width]

        # pos emb
        pos_embed = self.positional_embedding.to(x.dtype)  # [1+grid**2, width]
        cls_pos   = pos_embed[0].unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)       # [B,1,width]
        patch_pos = pos_embed[1:].unsqueeze(0).expand(batch_size, x.shape[1], -1)          # [B,grid**2,width]

        if self.add_cls_num > 0 and self.added_cls is not None:
            added_cls_tokens = (
                self.added_cls.to(x.dtype)
                .unsqueeze(0).expand(batch_size, self.add_cls_num, -1)                      # [B,add,width]
            )
            added_cls_pos = cls_pos.expand(batch_size, self.add_cls_num, -1)               # [B,add,width]
            added_cls_tokens = added_cls_tokens + added_cls_pos
        else:
            added_cls_tokens = x.new_empty(batch_size, 0, width)                            # [B,0,width]

        # apply pos to cls/patch
        cls_tokens = cls_tokens + cls_pos
        x_patch    = x + patch_pos

        # concat: [CLS], (optional added CLS), patches
        x = torch.cat([cls_tokens, added_cls_tokens, x_patch], dim=1)  # [B, 1+add+grid**2, width]
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        if isinstance(self.transformer, MoETransformer):
            x, _, router_logits = self.transformer((x, video_frame))
        else:
            x, video_frame = self.transformer((x, video_frame))
            router_logits = None

        x = x.permute(1, 0, 2)  # LND -> NLD

        return x, router_logits if router_logits is not None else x

class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_resolution: int,
        vision_layers: int,
        vision_width: int,
        vision_patch_size: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        linear_patch: str = '2d',
        args=None
    ):
    
        super().__init__()
        self.context_length = context_length
        
        self.use_moe = args.use_moe
        self.load_balancing_loss_coef = args.load_balancing_loss_coef


        vision_heads = vision_width // 64
        self.visual = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            linear_patch=linear_patch,
            args = args
        )
        
        if self.use_moe and len(args.text_moe_indices) > 0 :
            self.transformer = MoETransformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask,
                encoder_type = 'text',
                args=args
            )
        else:
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask,
            )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        if not self.use_moe:
            for block in self.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @staticmethod
    def get_config(pretrained_clip_name="ViT-B/32"):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViT-B-32.pt")
        if pretrained_clip_name in _MODELS and pretrained_clip_name in _PT_NAME:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[pretrained_clip_name])
        
        if pretrained_clip_name in ["ViT-B/32", "ViT-B/16"] and os.path.exists(model_path):
            pass
        else:
            if pretrained_clip_name in _MODELS:
                model_path = _download(_MODELS[pretrained_clip_name])
            elif os.path.isfile(pretrained_clip_name):
                model_path = pretrained_clip_name
            else:
                raise RuntimeError(f"Model {pretrained_clip_name} not found; available models = {available_models()}")

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        return state_dict

    def build_attention_mask(self, context_length):
        mask = torch.zeros(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_hidden=False,video_frame=-1):
        ### 이부분에서 router_logits 어떻게 반환되는지
        #import pdb;pdb.set_trace()
        out = self.visual(image.type(self.dtype),video_frame=video_frame)
        if isinstance(out, tuple):
            x, router_logits = out
        else:
            x, router_logits = out, None

        x = self.visual.ln_post(x) @ self.visual.proj
        image_features = x[:, 0, :]
        if self.use_moe:
            return image_features , router_logits
        return image_features

    def encode_text(self, text, return_hidden=False):
        x = self.token_embedding(text).type(self.dtype)
        pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype)
        x = x + pos_emd
        x = x.permute(1, 0, 2)
        transformer_out = self.transformer(x)
        if isinstance(transformer_out, tuple):
            x, video_frame = transformer_out[:2]  # for MoETransformer (x, video_frame, router_logits)
            router_logits = transformer_out[2]

        else:
            x, router_logits  = transformer_out, None
        x = x.permute(1, 0, 2)
        hidden = self.ln_final(x).type(self.dtype) @ self.text_projection
        text_features = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]
        if self.use_moe:
            return text_features,hidden, router_logits
        return text_features

    def forward(self, image, text):
        if self.use_moe:
            image_features , image_routerLogits = self.encode_image(image)
            text_features , text_routerLogits = self.encode_text(text)
        else:
            image_features = self.encode_image(image)
            text_features = self.encode_text(text)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict
    import pdb;pdb.set_trace()

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0] 
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = 224  # output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)

    
    return model.eval()

def build_attn_config(d_model, n_head, attn_drop=0.0):
    return SimpleNamespace(
        hidden_size=d_model,
        num_attention_heads=n_head,
        attention_dropout=attn_drop,
    )

