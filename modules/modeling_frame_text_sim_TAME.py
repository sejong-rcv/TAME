import logging

import torch
from torch import nn

from modules.until_module import PreTrainedModel, AllGather, CrossEn, MaxMarginRankingLoss, _split_in_proj, remap_state_for_clipattention
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip_moe_TAME import CLIP, convert_weights
from modules.MoE_utils import load_balancing_loss_func

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import os
import torchvision.utils as vutils
import torch.nn.functional as F
import math
logger = logging.getLogger(__name__)
allgather = AllGather.apply


class CTIA(nn.Module):
    def __init__(
        self,
        num_frames: int,
        kernel_size: int = 3,
        tau1: float = 1e-2,
        tau2: float = 1e-2,
        gaussian_sigma: float = 0.5,
        alpha_fixed=(1.0, 0.5, 0.5),   # (r, r_conv, r_graph)
        
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for 'same' padding."
        self.tau1 = tau1
        self.tau2 = tau2
        self.kernel_size = kernel_size
        self.gaussian_sigma = gaussian_sigma


        if isinstance(alpha_fixed, torch.Tensor):
            alpha_tensor = alpha_fixed.to(dtype=torch.float32).view(-1)
        else:
            alpha_tensor = torch.tensor(alpha_fixed, dtype=torch.float32).view(-1)
        assert alpha_tensor.numel() == 3, "alpha_fixed must have 3 values: (r, r_conv, r_graph)"
        
        self.alpha = nn.Parameter(alpha_tensor)      
        self.graph_strength = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.self_loop = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.graph_tau = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.num_frames = num_frames

    # ---------- Gaussian utilities ----------
    @staticmethod
    def _gaussian_kernel1d(ks: int, sigma: float, device=None, dtype=None):
        x = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        k = k / (k.sum() + 1e-8)
        return k.view(1, 1, ks)  # (out_ch=1, in_ch=1, K)

    def _gaussian_filter1d_masked(self, r: torch.Tensor, video_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply 1D Gaussian smoothing along the frame dimension with mask-aware normalization.

        Args:
            r:          [T, V, F] relevance over frames
            video_mask: [V, F], 1 = valid frame, 0 = padding

        Returns:
            [T, V, F]: mask-normalized, Gaussian-smoothed relevance.
        """
        T, V, Ff = r.shape
        k = self._gaussian_kernel1d(self.kernel_size, self.gaussian_sigma, r.device, r.dtype)
        pad = (self.kernel_size - 1) // 2

        # reshape to (T*V, 1, F) for a single conv1d call
        r_ = r.reshape(T * V, 1, Ff)
        m_ = video_mask.float().unsqueeze(1).expand(V, 1, Ff).repeat(T, 1, 1)  # (T*V, 1, F)

        # normalize by the sum of valid kernel weights
        num = F.conv1d(r_ * m_, k, padding=pad)
        den = F.conv1d(m_, k, padding=pad) + 1e-8
        r_conv = (num / den).reshape(T, V, Ff)

        # zero out padded frames
        r_conv = r_conv.masked_fill(video_mask.unsqueeze(0) == 0, 0.0)
        return r_conv

    @torch.no_grad()
    def _build_frame_graph(self, frame_features: torch.Tensor, video_mask: torch.Tensor) -> torch.Tensor:
        """
        Build a frame-wise affinity graph for each video.

        Args:
            frame_features: [V, F, D] frame embeddings (preferably L2-normalized)
            video_mask:     [V, F], 1 = valid frame, 0 = padding

        Returns:
            A: [V, F, F] row-stochastic adjacency matrix for each video.
        """
        V, Ff, D = frame_features.shape

        # approximate cosine similarity with dot product
        A = torch.einsum('vfd, vkd -> vfk', frame_features, frame_features)  # [V, F, F]

        pad_bool = (video_mask == 0)
        big_neg = torch.finfo(frame_features.dtype).min
        A = A.masked_fill(pad_bool.unsqueeze(-1), big_neg)
        A = A.masked_fill(pad_bool.unsqueeze(1), big_neg)

        # row-wise softmax normalization
        A = torch.softmax(A / torch.clamp(self.graph_tau, min=1e-4), dim=-1)

        # mix with self-loop
        eye = torch.eye(Ff, device=A.device, dtype=A.dtype).unsqueeze(0)  # [1, F, F]
        lam = torch.clamp(self.self_loop, 0.0, 1.0)
        A = lam * eye + (1.0 - lam) * A

        # re-normalize to be row-stochastic
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
        return A

    def forward(self, S: torch.Tensor, frame_features: torch.Tensor, video_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute sentence–video similarity via masked temporal filtering and graph propagation.

        Args:
            S:             [T, V, F] sentence–frame raw similarity
            frame_features:[V, F, D] frame embeddings
            video_mask:    [V, F], 1 = valid frame, 0 = padding

        Returns:
            logits: [T, V] sentence–video similarity scores.
        """
        T, V, Ff = S.shape
        assert Ff == self.num_frames, f"Expected num_frames {self.num_frames}, got {Ff}"

        pad_bool = (video_mask == 0)
        big_neg = torch.finfo(S.dtype).min
        mask_tvf = pad_bool.unsqueeze(0).expand(T, -1, -1)

        # 1) Initial relevance r
        S_masked = S.masked_fill(mask_tvf, big_neg)
        r = torch.softmax(S_masked / self.tau1, dim=-1)
        r = r.masked_fill(mask_tvf, 0.0)

        # 2) Local temporal mixing (Gaussian smoothing with mask normalization)
        r_conv = self._gaussian_filter1d_masked(r, video_mask)

        # 3) Non-local frame graph propagation
        A = self._build_frame_graph(frame_features, video_mask)
        r_graph = torch.einsum('vff, tvf -> tvf', A, r)
        r_graph = r_graph.masked_fill(mask_tvf, 0.0)
        
        # 4) Fusion and final softmax over frames
        alpha = torch.relu(self.alpha)  # (3,) = [a_r, a_conv, a_graph]
        u = alpha[0] * r + alpha[1] * r_conv + alpha[2] * (self.graph_strength * r_graph)
        u = u.masked_fill(mask_tvf, big_neg)
        
        w = torch.softmax(u / self.tau2, dim=-1).masked_fill(mask_tvf, 0.0)
        logits = (w * S).sum(dim=-1)
        return logits

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs,use_moe=False, visual_MoE_args=None, text_MoE_args=None, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):
        
        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        model = cls(cross_config, clip_state_dict, *inputs, **kwargs,)

        state_dict = remap_state_for_clipattention(model, state_dict, prefix="clip.visual.transformer.resblocks")
        state_dict = remap_state_for_clipattention(model, state_dict, prefix="clip.transformer.resblocks")
        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        ## <=== End of initialization trick

        if hasattr(model, "clip") and hasattr(model.clip, "visual") and hasattr(model.clip.visual, "vip_embeddings"):

            vip = getattr(getattr(getattr(model, "clip", None), "visual", None), "vip_embeddings", None)

            k_cls_src   = "clip.visual.class_embedding"
            k_cls_tgt   = "clip.visual.vip_embeddings.class_embedding"
            k_patch_src = "clip.visual.conv1.weight"
            k_patch_tgt = "clip.visual.vip_embeddings.patch_embedding.weight"
            k_pos_src   = "clip.visual.positional_embedding"
            k_pos_tgt   = "clip.visual.vip_embeddings.position_embedding.weight"

            if k_cls_src in state_dict and hasattr(vip, "position_embedding"):
                
                state_dict[k_cls_tgt] = state_dict[k_cls_src].clone()

            if k_patch_src in state_dict and hasattr(vip, "patch_embedding"):
                state_dict[k_patch_tgt] = state_dict[k_patch_src].clone()

            if k_pos_src in state_dict and hasattr(vip, "position_embedding"):
                state_dict[k_pos_tgt] = state_dict[k_pos_src]                    


        if task_config.use_moe : 
            
            if state_dict is not None:
                model = cls.load_weights_for_moe_vip(model, state_dict,task_config = task_config)
        else: 
            
            if state_dict is not None:
                model = cls.init_preweight(model, state_dict, task_config=task_config)
        
        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        
        self.task_config = task_config 
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))
        
        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch,
            args = task_config
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

        num_words = task_config.max_words
        num_frames = self.task_config.max_frames

        self.use_original_clip_for_frame_features = True    

        self.global_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)
        self.global_mat_weight_1 = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)

        a = list(map(float, self.task_config.ctia_alpha.split(',')))
        self.ctia = CTIA(num_frames=self.task_config.max_frames, kernel_size=3, tau1=1e-2, tau2=1e-2,alpha_fixed=a)

        num_frames = self.task_config.max_frames

        self.loss_fct = CrossEn()
        self.frame_match_weight = 1.0

        self.use_load_balancing_loss = task_config.use_load_balancing_loss
        self.load_balancing_loss_coef = task_config.load_balancing_loss_coef

        self.use_hard_negative = getattr(task_config, "use_hard_negative", False)
        self.hard_neg_margin   = float(getattr(task_config, "hard_neg_margin", 0.2))

        self.use_sentence_frame_logits = not bool(getattr(task_config, "no_sentence_frame_logits", False))
        self.not_use_global_mat_weight = bool(getattr(task_config, "not_use_global_mat_weight", False))
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None,return_router_logits=False):
        
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w) 
        video_frame = bs * ts 
        
        sequence_output, visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         video, video_mask, shaped=True, video_frame=video_frame, return_router_logits = return_router_logits)
        
        if return_router_logits:
            (sequence_output, seq_features), seq_router_logits = sequence_output
            visual_output , vis_router_logits  = visual_output

        if self.training:
            loss = 0.
            
            sim_matrix = self.get_similarity_logits(sequence_output, seq_features, visual_output, attention_mask, 
                                        video_mask, shaped=True, loose_type=self.loose_type)
            
 
            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss_semantic = (sim_loss1 + sim_loss2) / 2
            loss = loss + sim_loss_semantic
            
            seq_load_balancing_loss = torch.tensor(0.0, device=sim_loss1.device)
            vis_load_balancing_loss = torch.tensor(0.0, device=sim_loss1.device)

            if self.use_load_balancing_loss :
                    seq_load_balancing_loss = self.load_balancing_loss_coef * load_balancing_loss_func(seq_router_logits,num_experts = self.task_config.num_experts, top_k=self.task_config.top_k)
                    loss += seq_load_balancing_loss
                    
                    vis_load_balancing_loss = self.load_balancing_loss_coef * load_balancing_loss_func(vis_router_logits,num_experts = self.task_config.num_experts, top_k=self.task_config.vis_top_k)
                    loss += vis_load_balancing_loss
                        

            hard_negative_loss = torch.tensor(0.0, device=sim_matrix.device)
            if self.use_hard_negative:
                # --- hard negative loss ---
               

                bs = sim_matrix.size(0)
                labels = torch.arange(bs, device=sim_matrix.device)

                logits_clone = sim_matrix.clone()
                logits_clone[range(bs), labels] = float('-inf')

                hard_neg_logits, _ = logits_clone.max(dim=1)   # [bs]
                pos_logits = sim_matrix[range(bs), labels] # [bs]

                margin = self.hard_neg_margin
                hard_negative_loss = torch.clamp(margin + hard_neg_logits - pos_logits, min=0).mean()
                loss = loss + hard_negative_loss
   
            return {
                "total_loss": loss,
                "sim_loss_global": sim_loss_semantic,
                "seq_load_balancing_loss": seq_load_balancing_loss,
                "vis_load_balancing_loss": vis_load_balancing_loss,
                "hard_loss": hard_negative_loss
            }
        else:
            return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False, return_router_logits=False):
        
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        
        bs_pair = input_ids.size(0)
       
        result = self.clip.encode_text(input_ids, return_hidden=True)
        if self.clip.use_moe:
            sequence_hidden,seq_features , router_logits = result
        else:
            sequence_hidden, router_logits = result, 0.0

        sequence_hidden, seq_features = sequence_hidden.float(), seq_features.float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))
        if return_router_logits:
            return sequence_hidden,seq_features, router_logits 
        return sequence_hidden, seq_features
  
    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1, return_router_logits=False):
        
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts
        
        bs_pair = video_mask.size(0)
        result = self.clip.encode_image(video,return_hidden=True, video_frame=video_frame)
        if self.clip.use_moe:
            visual_hidden ,router_logits = result
        else:
            visual_hidden, router_logits = result, 0.0
        
        visual_hidden = visual_hidden.float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
 

        if return_router_logits:
            return visual_hidden, router_logits 
        return visual_hidden

    def get_sequence_visual_output( self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1, return_router_logits=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_result = self.get_sequence_output(
            input_ids, token_type_ids, attention_mask,
            shaped=True, return_router_logits=return_router_logits
        )
        visual_result = self.get_visual_output(
            video, video_mask, shaped=True, video_frame=video_frame,
            return_router_logits=return_router_logits
        )

        if return_router_logits:
            sequence_output,seq_features, seq_router_logits  = sequence_result
            visual_output , vis_router_logits = visual_result
            return ((sequence_output,seq_features), seq_router_logits), (visual_output, vis_router_logits)
        else:
            sequence_output,seq_features = sequence_result
            visual_output = visual_result
            return (sequence_output, seq_features), visual_output 

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _loose_similarity(self, sequence_output, seq_features, visual_output, attention_mask, video_mask, sim_header="meanP"):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        
        loss = 0.

        if sim_header == "meanP":
            # Default: Parameter-free type
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original

        # video-level visual feature 
        video_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        video_output = self._mean_pooling_for_similarity_visual(video_output, video_mask)
        video_output = video_output / video_output.norm(dim=-1, keepdim=True)                    # [bs, dim]

        # frame-level visual features       
        if self.use_original_clip_for_frame_features:
            frame_features = visual_output_original / visual_output_original.norm(dim=-1, keepdim=True)                # [bs, num_frames, dim]
        else:
            frame_features = visual_output / visual_output.norm(dim=-1, keepdim=True)                                  # [bs, num_frames, dim]

        # sentence-level textual feature
        sentence_output = sequence_output.squeeze(1)
        sentence_output  = sentence_output / sentence_output.norm(dim=-1, keepdim=True)          # [bs, dim]

        logit_scale = self.clip.logit_scale.exp()

        if self.training:
            video_output = allgather(video_output, self.task_config)
            frame_features = allgather(frame_features, self.task_config)
            sentence_output = allgather(sentence_output, self.task_config)
            attention_mask = allgather(attention_mask, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            #torch.distributed.barrier()

        
        if self.not_use_global_mat_weight:
            video_sentence_logits = logit_scale * torch.matmul(sentence_output, video_output.t())
        else : 
            video_sentence_logits = logit_scale * torch.matmul(torch.matmul(sentence_output, self.global_mat_weight), torch.matmul(video_output,self.global_mat_weight_1).t() )
        # sentence-frame score
        if self.use_sentence_frame_logits:
            # sentence-frame score with CTIA
            S_tf = torch.einsum('td, vfd -> tvf', sentence_output, frame_features)  # [T,V,F]
            sentence_frame_logits = logit_scale * self.ctia(S_tf, frame_features, video_mask)  # [T,V]
            logits = (video_sentence_logits + sentence_frame_logits) / 2.0
        else:
            # CTIA ablation: video-only
            logits = video_sentence_logits
        return logits


    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text      # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1)\
            .to(device=attention_mask.device, dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, seq_features, visual_output, attention_mask, video_mask, shaped=False, loose_type=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        # contrastive_direction = ()
        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf"]
            retrieve_logits = self._loose_similarity(sequence_output, seq_features, visual_output, attention_mask, video_mask, sim_header=self.sim_header)
        else:
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask, )

        return retrieve_logits  #, sim_matrix_semantic #, contrastive_direction
    
    





