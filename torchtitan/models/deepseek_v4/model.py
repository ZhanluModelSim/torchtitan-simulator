# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear

from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleDict

from .moe import MoE, MoEArgs

logger = logging.getLogger()


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    original_dtype = x.dtype
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.to(original_dtype)


def hadamard_transform_ref(x, hadamard_mat, scale=1.0):
    """
    Eager implementation of the Hadamard transform
    Args:
        x:(torch.Tensor): input tensor
    """

    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, hadamard_mat)
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def rotate_activation(x: torch.Tensor, hadamard_mat: torch.Tensor) -> torch.Tensor:
    hidden_size = x.size(-1)
    return hadamard_transform_ref(x, hadamard_mat, scale=hidden_size**-0.5)


class RMSNorm(Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.eps = config.eps
        # rmsnorm in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        self.weight = nn.Parameter(torch.ones(config.dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


class Compressor(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        args: "DeepSeekV4Model.Config"
        compress_ratio: int = 4
        head_dim: int = 512
        rotate: bool = False

    def __init__(self, config: Config):
        super().__init__()
        args = config.args
        self.dim = args.dim
        self.head_dim = config.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = config.head_dim - args.rope_head_dim
        self.compress_ratio = config.compress_ratio
        self.overlap = config.compress_ratio == 4
        self.rotate = config.rotate
        coff = 1 + self.overlap
        self.ape = nn.Parameter(
            torch.empty(
                config.compress_ratio, coff * self.head_dim, dtype=torch.float32
            )
        )
        # wkv and wgate must stay in fp32: ``Compressor.forward`` upcasts ``x``
        # to fp32 and the downstream ``score.softmax(dim=2)`` is numerically
        # fragile in bf16. ``Linear.Config`` does not expose a dtype field so we
        # cast the parameter to fp32 after build.
        self.wkv = Linear.Config(
            in_features=self.dim,
            out_features=coff * self.head_dim,
            bias=False,
        ).build()
        self.wgate = Linear.Config(
            in_features=self.dim,
            out_features=coff * self.head_dim,
            bias=False,
        ).build()
        self.wkv.to(torch.float32)
        self.wgate.to(torch.float32)
        self.norm = RMSNorm.Config(dim=self.head_dim, eps=args.norm_eps).build()
        # If overlap is enabled, state[:, :ratio] for overlapping compression and state[:, ratio:] for normal compression.

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        # tensor: [b,s,r,2d]
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        _, seqlen, _ = x.size()
        ratio, overlap = self.compress_ratio, self.overlap
        dtype = x.dtype
        if seqlen < ratio:
            return
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        freqs_cis = freqs_cis[:cutoff:ratio]
        if remainder > 0:
            kv, _ = kv.split([cutoff, remainder], dim=1)
            score = score[:, :cutoff]

        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape
        if overlap:
            kv = self.overlap_transform(kv, 0)
            score = self.overlap_transform(score, float("-inf"))

        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        kv = self.norm(kv.to(dtype))
        kv_rot = apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)
        kv = torch.cat([kv[..., : -self.rope_head_dim], kv_rot], dim=-1)
        return kv

    def init_weights(self, init_std: float):
        linear_list = [
            self.wkv,
            self.wgate,
        ]
        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.ape, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.norm.weight, mean=1, std=0.02)


class Indexer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        args: "DeepSeekV4Model.Config"
        compress_ratio: int = 4

    def __init__(self, config: Config):
        super().__init__()
        args = config.args
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = Linear.Config(
            in_features=self.q_lora_rank,
            out_features=self.n_heads * self.head_dim,
            bias=False,
        ).build()
        self.weights_proj = Linear.Config(
            in_features=self.dim,
            out_features=self.n_heads,
            bias=False,
        ).build()
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = config.compress_ratio
        self.compressor = Compressor.Config(
            args=args,
            compress_ratio=config.compress_ratio,
            head_dim=self.head_dim,
            rotate=True,
        ).build()

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        freqs_cis: torch.Tensor,
        hadamard_mat: torch.Tensor,
        offset: int,
    ):
        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q = q.clone()
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
        q_rope = apply_rotary_emb(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)
        q = rotate_activation(q, hadamard_mat)
        k = self.compressor(x, freqs_cis)
        k = rotate_activation(k, hadamard_mat)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        return q, k, weights

    def init_weights(self, init_std: float):
        linear_list = [self.wq_b, self.weights_proj]
        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        self.compressor.init_weights(init_std)


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for DSA indexer loss."""

    # pyrefly: ignore [bad-assignment]
    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the indexer_loss by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The indexer loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for indexer loss.

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled indexer loss
                                               gradient.
        """
        (loss,) = ctx.saved_tensors
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            # pyrefly: ignore [bad-assignment]
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=loss.device
            )
        dsa_indexer_loss_backward_scale = (
            DSAIndexerLossAutoScaler.main_loss_backward_scale
        )
        scaled_dsa_indexer_loss_grad = (
            torch.ones_like(loss) * dsa_indexer_loss_backward_scale
        )
        return grad_output, scaled_dsa_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the indexer loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            # pyrefly: ignore [bad-assignment]
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


class DSAIndexerLossLoggingHelper:
    """Helper class for logging DSAIndexer losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
    ):
        """Save the DSA indexer loss for logging.
        Args:
            loss (torch.Tensor): The loss tensor.
            layer_number (int): Layer index of the loss.
            num_layers (int): The number of total layers.
        """
        # Skip DSA indexer loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_layers, device=loss.device)
        tracker["values"][layer_number - 1] += (
            loss.to_local().detach() if isinstance(loss, DTensor) else loss.detach()
        )

    @staticmethod
    def clean_loss_in_tracker():
        """Clear the DSA indexer losses."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        tracker["values"].zero_()

    @staticmethod
    def track_dsa_indexer_metrics(total_acc_steps: int):
        """Track the DSA Indexer metrics for logging."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        das_indexer_losses = tracker["values"]
        das_indexer_num_layers = das_indexer_losses.shape[0]
        loss = das_indexer_losses.sum() / das_indexer_num_layers / total_acc_steps
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        logger.info(f"indexer loss: {loss.item()}")


class DSAIndexerLoss(Module):
    """Compute dsa indexer loss at sparse training stage
    Reference: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf
    Args:
        selected_main_attn_dist: Q dist
        index_score: P dist
        topk_indices: Selected top-K indices for sparse phase
        loss_scale: Dsa indexer loss scale
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(
        self,
        selected_main_attn_dist,
        index_score,
        topk_indices,
        loss_scale,
    ):

        index_score = F.softmax(index_score, dim=-1, dtype=torch.float32)

        # considering only the selected token
        selected_main_attn_dist = F.normalize(selected_main_attn_dist, p=1, dim=-1)
        loss = (
            F.kl_div(
                (index_score + 1e-10).log(),
                selected_main_attn_dist + 1e-10,
                reduction="none",
            )
            .sum(dim=-1)
            .mean()
        )
        loss *= loss_scale

        return loss


class GetAttnScores(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(
        self,
        query,
        key,
        attention_mask,
        num_attn_head_per_group,
        attn_scale,
    ):
        """aggregate the main attention scores"""
        if num_attn_head_per_group > 1:
            key = key.repeat_interleave(num_attn_head_per_group, dim=1)

        num_head_q = query.shape[1]
        num_head_k = key.shape[1]
        if num_head_q != num_head_k and num_head_k != 1:
            raise NotImplementedError(
                f"Only support num_head_q == num_head_k or num_head_k == 1. "
                f"Current {num_head_q=}, {num_head_k=}."
            )

        attn = (query @ key.transpose(-1, -2)) * attn_scale

        if attention_mask is not None:
            attn.masked_fill_(attention_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1, dtype=torch.float32)
        attn = attn.sum(dim=1)
        return attn


class LiLoss(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        n_heads: int
        softmax_scale: float
        compress_ratio: int
        layer_id: int
        n_layers: int

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.softmax_scale = config.softmax_scale
        self.compress_ratio = config.compress_ratio
        self.get_attn_scores = GetAttnScores.Config().build()
        self.compute_dsa_indexer_loss = DSAIndexerLoss.Config().build()
        self.layer_id = config.layer_id
        self.n_layers = config.n_layers

    def save_loss(self, loss):
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss, self.layer_id, self.n_layers
        )

    def forward(
        self,
        q,
        kv_compress,
        q_indexer,
        k_indexer,
        weights,
        compress_topk_idxs,
        index_score,
        attention_masks,
        offset,
    ):
        compress_topk_idxs = torch.where(
            compress_topk_idxs == -1, compress_topk_idxs, compress_topk_idxs - offset
        )
        main_attn_dist = self.get_attn_scores(
            q.transpose(1, 2).detach(),
            kv_compress.unsqueeze(1).detach(),
            attention_masks,
            self.n_heads,
            self.softmax_scale,
        )
        selected_main_attn_dist = torch.gather(
            main_attn_dist, dim=-1, index=compress_topk_idxs
        )
        loss = self.compute_dsa_indexer_loss(
            selected_main_attn_dist,
            index_score,
            compress_topk_idxs,
            self.softmax_scale,
        )
        self.save_loss(loss)
        return loss


class ComputeIndexerLoss(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        n_heads: int
        softmax_scale: float
        compress_ratio: int
        layer_id: int
        n_layers: int

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.li_loss = LiLoss.Config(
            n_heads=config.n_heads,
            softmax_scale=config.softmax_scale,
            compress_ratio=config.compress_ratio,
            layer_id=config.layer_id,
            n_layers=config.n_layers,
        ).build()

    def forward(
        self,
        compress_topk_idxs,
        offset,
        q,
        kv_compress,
        attention_masks,
        index_score,
        q_indexer,
        k_indexer,
        weights,
    ):
        return self.li_loss(
            q,
            kv_compress,
            q_indexer,
            k_indexer,
            weights,
            compress_topk_idxs,
            index_score,
            attention_masks,
            offset,
        )


class GetWindowTopkIdxs(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(self, window_size: int, bsz: int, seqlen: int):
        base = torch.arange(seqlen).unsqueeze(1)
        window_topk = (base - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size)
        )
        # Use seqlen as the "invalid" sentinel instead of -1 because
        # newer PyTorch validates scatter_ indices strictly.
        window_topk = torch.where(window_topk > base, seqlen, window_topk)
        return window_topk.unsqueeze(0).expand(bsz, -1, -1)


class GetCompressTopkIdxs(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        ratio: int = 1

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.ratio = config.ratio

    def forward(self, x: torch.Tensor, offset: int):
        bsz, seqlen = x.shape[0], x.shape[1]
        matrix = torch.arange(seqlen // self.ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // self.ratio
        # Use offset + seqlen as sentinel (maps to the attention sink slot),
        # avoiding -1 which triggers bounds errors in newer PyTorch scatter.
        compress_topk = torch.where(mask, offset + seqlen, matrix + offset)
        return compress_topk.unsqueeze(0).expand(bsz, -1, -1)


def precompute_freqs_cis(
    model_args: DeepSeekV4Model.Config, with_compressor: bool
) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = model_args.rope_head_dim
    seqlen = model_args.max_seq_len
    # disable YaRN and use base rope_theta in pure sliding-window attention
    original_seq_len = model_args.original_seq_len if with_compressor else 0
    base = model_args.compress_rope_theta if with_compressor else model_args.rope_theta
    factor = model_args.rope_factor
    beta_fast = model_args.beta_fast
    beta_slow = model_args.beta_slow

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


class SparseAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_id: int
        args: "DeepSeekV4Model.Config"

    def __init__(self, config: Config) -> None:
        super().__init__()
        layer_id = config.layer_id
        args = config.args
        self.layer_id = layer_id
        self.window_size = args.window_size
        self.rd = args.rope_head_dim
        self.compress_ratio = (
            args.compress_ratios[layer_id]
            if layer_id < args.n_layers
            else args.mtp_layer_compress_ratio
        )
        self.softmax_scale = args.head_dim**-0.5
        self.get_window_topk_idxs = GetWindowTopkIdxs.Config().build()
        self.get_compress_topk_idxs = GetCompressTopkIdxs.Config(
            ratio=self.compress_ratio
        ).build()

    def forward(
        self,
        query_states: torch.Tensor,
        kv_states: torch.Tensor,
        attn_sink: torch.Tensor,
        kv_compress: torch.Tensor | None = None,
        compress_topk_idxs: torch.Tensor | None = None,
    ):
        bsz, seqlen, _, _ = query_states.size()
        topk_idxs = self.get_window_topk_idxs(self.window_size, bsz, seqlen)
        if self.compress_ratio > 1:
            offset = kv_states.size(1)
            if compress_topk_idxs is None:
                compress_topk_idxs = self.get_compress_topk_idxs(query_states, offset)
            topk_idxs = torch.cat(
                [
                    topk_idxs.to(kv_states.device),
                    compress_topk_idxs.to(kv_states.device),
                ],
                dim=-1,
            )
        topk_idxs = topk_idxs.int()

        if self.compress_ratio > 1:
            if kv_compress is not None:
                kv_states = torch.cat([kv_states, kv_compress], dim=1)

        query_states = query_states.transpose(1, 2)
        kv_states = kv_states.unsqueeze(1)
        attn_weights = (
            torch.matmul(query_states, kv_states.transpose(2, 3)) * self.softmax_scale
        )
        topk_idxs = topk_idxs.to(query_states.device)
        index_mask = torch.full(
            (query_states.shape[0], 1, query_states.shape[2], kv_states.shape[2] + 1),
            fill_value=torch.finfo(torch.bfloat16).min,
            dtype=torch.bfloat16,
            device="cpu",
        ).scatter_(-1, topk_idxs.unsqueeze(1), 0)

        attn_weights = attn_weights + index_mask[..., :-1]
        sinks = attn_sink.reshape(1, -1, 1, 1).expand(
            query_states.shape[0], -1, query_states.shape[-2], -1
        )
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits = (
            combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        )
        probs = nn.functional.softmax(
            combined_logits.float(), dim=-1, dtype=combined_logits.dtype
        )
        scores = probs[..., :-1]
        attn_output = torch.matmul(scores, kv_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output


class LiCompute(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        ratio: int
        index_topk: int

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.ratio = config.ratio
        self.index_topk = config.index_topk

    def forward(
        self,
        q_indexer: torch.Tensor,
        k_indexer: torch.Tensor,
        weights: torch.Tensor,
        seqlen: int,
        offset: int,
    ):
        end_pos = seqlen
        # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
        index_score = torch.einsum("bshd,btd->bsht", q_indexer, k_indexer)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        device = index_score.device
        base = torch.arange(seqlen, device=device).unsqueeze(1)
        mask = (
            torch.arange(seqlen // self.ratio, device=device).unsqueeze(0)
            >= (base + 1) // self.ratio
        )
        index_score += torch.where(mask, torch.finfo(q_indexer.dtype).min, 0)
        index_score, topk_idxs = index_score.topk(
            min(self.index_topk, end_pos // self.ratio), dim=-1
        )
        mask = topk_idxs >= (base + 1) // self.ratio
        compress_topk_idxs = torch.where(mask, offset + seqlen, topk_idxs + offset)
        return compress_topk_idxs, index_score


class PreAttention(Module):
    """Pre-attention module: projection layers for MLA attention."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_id: int
        args: "DeepSeekV4Model.Config"

    def __init__(self, config: Config):
        super().__init__()
        layer_id = config.layer_id
        args = config.args
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.eps = args.norm_eps
        self.compress_ratio = (
            args.compress_ratios[layer_id]
            if layer_id < args.n_layers
            else args.mtp_layer_compress_ratio
        )
        self.use_sfa = args.use_sfa

        self.wq_a = Linear.Config(
            in_features=args.dim,
            out_features=self.q_lora_rank,
            bias=False,
        ).build()
        self.q_norm = RMSNorm.Config(dim=self.q_lora_rank, eps=self.eps).build()
        self.wq_b = Linear.Config(
            in_features=self.q_lora_rank,
            out_features=self.n_heads * self.head_dim,
            bias=False,
        ).build()
        self.wkv = Linear.Config(
            in_features=args.dim,
            out_features=self.head_dim,
            bias=False,
        ).build()
        self.kv_norm = RMSNorm.Config(dim=self.head_dim, eps=self.eps).build()
        if self.compress_ratio == 4:
            self.compressor = Compressor.Config(
                args=args, compress_ratio=self.compress_ratio, head_dim=self.head_dim
            ).build()
            self.indexer = Indexer.Config(
                args=args, compress_ratio=self.compress_ratio
            ).build()
        elif self.compress_ratio > 1:
            self.compressor_128 = Compressor.Config(
                args=args, compress_ratio=self.compress_ratio, head_dim=self.head_dim
            ).build()

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, hadamard_mat: torch.Tensor
    ):
        rd = self.rope_head_dim
        # Q projection
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
        q_rope = apply_rotary_emb(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)

        kv = kv_compress = q_indexer = k_indexer = weights = offset = None

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)
        kv_rope = apply_rotary_emb(kv_rope, freqs_cis)
        kv = torch.cat([kv_nope, kv_rope], dim=-1)
        offset = 0 if self.use_sfa else kv.size(1)

        if self.compress_ratio > 1 and hasattr(self, "indexer"):
            q_indexer, k_indexer, weights = self.indexer(
                x.detach(), qr.detach(), freqs_cis, hadamard_mat, offset
            )

        if self.compress_ratio == 4:
            kv_compress = self.compressor(x, freqs_cis)
        elif self.compress_ratio > 1:
            kv_compress = self.compressor_128(x, freqs_cis)

        return q, kv, kv_compress, q_indexer, k_indexer, weights, offset

    def init_weights(self, init_std: float):
        linear_list = [self.wq_a, self.wq_b]
        if hasattr(self, "wkv"):
            linear_list.append(self.wkv)
        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        if hasattr(self, "kv_norm"):
            nn.init.trunc_normal_(self.kv_norm.weight, mean=1, std=0.02)
        nn.init.trunc_normal_(self.q_norm.weight, mean=1, std=0.02)
        if self.compress_ratio == 4:
            self.indexer.init_weights(init_std)
            self.compressor.init_weights(init_std)
        elif self.compress_ratio > 1:
            self.compressor_128.init_weights(init_std)


class InnerAttention(Module):
    """Inner attention module: sparse attention and indexer ops."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_id: int
        args: "DeepSeekV4Model.Config"

    def __init__(self, config: Config):
        super().__init__()
        layer_id = config.layer_id
        args = config.args
        self.compress_ratio = (
            args.compress_ratios[layer_id]
            if layer_id < args.n_layers
            else args.mtp_layer_compress_ratio
        )
        self.attn_sink = nn.Parameter(torch.empty(args.n_heads, dtype=torch.float32))
        self.sparse_attn = SparseAttention.Config(layer_id=layer_id, args=args).build()
        if self.compress_ratio == 4:
            self.li_compute = LiCompute.Config(
                ratio=self.compress_ratio, index_topk=args.index_topk
            ).build()

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor | None,
        kv_compress: torch.Tensor | None,
        q_indexer: torch.Tensor | None,
        k_indexer: torch.Tensor | None,
        weights: torch.Tensor | None,
        seqlen: int,
        offset: int | None,
    ):
        compress_topk_idxs = index_score = None
        if (
            self.compress_ratio > 1
            and hasattr(self, "li_compute")
            and q_indexer is not None
        ):
            compress_topk_idxs, index_score = self.li_compute(
                q_indexer, k_indexer, weights, seqlen, offset
            )

        # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
        o = self.sparse_attn(q, kv, self.attn_sink, kv_compress, compress_topk_idxs)
        return o, compress_topk_idxs, index_score


class PostAttention(Module):
    """Post-attention module: output projection following attention."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        args: "DeepSeekV4Model.Config"

    def __init__(self, config: Config):
        super().__init__()
        args = config.args
        self.n_heads = args.n_heads
        self.o_lora_rank = args.o_lora_rank
        self.n_groups = args.o_groups
        self.rope_head_dim = args.rope_head_dim
        self.head_dim = args.head_dim
        self.wo_a = Linear.Config(
            in_features=self.n_heads * self.head_dim // self.n_groups,
            out_features=self.n_groups * self.o_lora_rank,
            bias=False,
        ).build()
        self.wo_b = Linear.Config(
            in_features=self.n_groups * self.o_lora_rank,
            out_features=args.dim,
            bias=False,
        ).build()

    def forward(
        self,
        o: torch.Tensor,
        freqs_cis: torch.Tensor,
        bsz: int,
        seqlen: int,
        n_local_groups: int,
    ):
        rd = self.rope_head_dim
        o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
        o_rope = apply_rotary_emb(o_rope, freqs_cis, True)
        o = torch.cat([o_nope, o_rope], dim=-1)
        o = o.view(bsz, seqlen, n_local_groups, -1)
        wo_a = self.wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.reshape(bsz, seqlen, -1))

    def init_weights(self, init_std: float):
        for linear in [self.wo_a, self.wo_b]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)


class Attention(Module):
    """Multi-Query Attention (MQA) Layer."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_id: int
        args: "DeepSeekV4Model.Config"

    def __init__(self, config: Config):
        super().__init__()
        layer_id = config.layer_id
        args = config.args
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers + args.num_mtp_modules
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.compress_ratio = (
            args.compress_ratios[layer_id]
            if layer_id < args.n_layers
            else args.mtp_layer_compress_ratio
        )
        self.args = args

        self.pre_attention = PreAttention.Config(layer_id=layer_id, args=args).build()
        self.inner_attention = InnerAttention.Config(
            layer_id=layer_id, args=args
        ).build()
        self.post_attention = PostAttention.Config(args=args).build()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        hadamard_mat: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ):
        bsz, seqlen, _ = x.size()
        freqs_cis = freqs_cis[:seqlen].to(x.device)

        q, kv, kv_compress, q_indexer, k_indexer, weights, offset = self.pre_attention(
            x, freqs_cis, hadamard_mat
        )

        n_local_groups = self.n_groups // (self.n_heads // q.shape[2])

        o, compress_topk_idxs, index_score = self.inner_attention(
            q, kv, kv_compress, q_indexer, k_indexer, weights, seqlen, offset
        )

        x = self.post_attention(o, freqs_cis, bsz, seqlen, n_local_groups)

        return (
            x,
            compress_topk_idxs,
            offset,
            q,
            kv_compress,
            attention_masks,
            index_score,
            q_indexer,
            k_indexer,
            weights,
        )

    def init_weights(self, init_std: float, buffer_device):
        self.pre_attention.init_weights(init_std)
        nn.init.trunc_normal_(self.inner_attention.attn_sink, mean=0.0, std=0.02)
        self.post_attention.init_weights(init_std)


class HcSplitSinkhorn(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(
        self,
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
        comb = comb.unflatten(-1, (hc_mult, hc_mult))

        pre = (
            F.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].unsqueeze(0).unsqueeze(0))
            + eps
        )
        post = 2 * F.sigmoid(
            post * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult].unsqueeze(0).unsqueeze(0)
        )
        comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].view(
            hc_mult, hc_mult
        ).unsqueeze(0).unsqueeze(0)

        comb = comb.softmax(-1) + eps
        col_sum = comb.sum(-2, keepdim=True)
        comb = comb / (col_sum + eps)
        for _ in range(sinkhorn_iters - 1):
            row_sum = comb.sum(-1, keepdim=True)
            comb = comb / (row_sum + eps)
            col_sum = comb.sum(-2, keepdim=True)
            comb = comb / (col_sum + eps)
        return pre, post, comb


class HcPost(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        return y.type_as(x)


class HcPre(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_mult: int
        hc_sinkhorn_iters: int
        hc_eps: float
        norm_eps: float

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.norm_eps
        self.torch_hc_split_sinkhorn = HcSplitSinkhorn.Config().build()

    def forward(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre, post, comb = self.torch_hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb


class DeepSeekV4TransformerBlock(Module):
    """
    Transformer block with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_id: int
        model_args: "DeepSeekV4Model.Config"

    def __init__(self, config: Config):
        super().__init__()
        layer_id = config.layer_id
        model_args = config.model_args
        self.moe_enabled = True
        self.layer_id = layer_id
        self.norm_eps = model_args.norm_eps
        self.attention = Attention.Config(layer_id=layer_id, args=model_args).build()
        self.moe = MoE.Config(
            moe_args=model_args.moe_args,
            dim=model_args.dim,
            hidden_dim=model_args.moe_inter_dim,
            layer_id=layer_id,
            vocab_size=model_args.vocab_size,
        ).build()
        self.attention_norm = RMSNorm.Config(
            dim=model_args.dim, eps=self.norm_eps
        ).build()
        self.ffn_norm = RMSNorm.Config(dim=model_args.dim, eps=self.norm_eps).build()
        self.hc_mult = hc_mult = model_args.hc_mult
        self.hc_sinkhorn_iters = model_args.hc_sinkhorn_iters
        self.hc_eps = model_args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * model_args.dim
        origin_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.empty(3))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3))
        torch.set_default_dtype(origin_dtype)
        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.hc_post = HcPost.Config().build()
        self.hc_pre = HcPre.Config(
            hc_mult=self.hc_mult,
            hc_sinkhorn_iters=self.hc_sinkhorn_iters,
            hc_eps=self.hc_eps,
            norm_eps=self.norm_eps,
        ).build()
        self.compress_ratio = (
            model_args.compress_ratios[layer_id]
            if layer_id < model_args.n_layers
            else model_args.mtp_layer_compress_ratio
        )
        self.cal_index_loss = ComputeIndexerLoss.Config(
            n_heads=model_args.n_heads,
            softmax_scale=model_args.head_dim**-0.5,
            compress_ratio=self.compress_ratio,
            layer_id=layer_id,
            n_layers=model_args.n_layers,
        ).build()

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        freqs_cis: torch.Tensor,
        hadamard_mat: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hc_mult, dim).
            input_ids (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attention_norm(x)
        (
            x,
            compress_topk_idxs,
            offset,
            q,
            kv_compress,
            attention_masks,
            index_score,
            q_indexer,
            k_indexer,
            weights,
        ) = self.attention(x, freqs_cis, hadamard_mat, attention_masks)
        if self.attention.compress_ratio > 1 and hasattr(
            self.attention.pre_attention, "indexer"
        ):
            loss = self.cal_index_loss(
                compress_topk_idxs,
                offset,
                q,
                kv_compress,
                attention_masks,
                index_score,
                q_indexer,
                k_indexer,
                weights,
            )
            x = DSAIndexerLossAutoScaler.apply(x, loss)

        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.moe(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            nn.init.trunc_normal_(norm.weight, mean=1, std=0.02)
        self.attention.init_weights(self.weight_init_std, buffer_device)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        if self.hc_ffn_fn is not None:
            nn.init.trunc_normal_(self.hc_ffn_fn, mean=0.0, std=0.02)
        if self.hc_ffn_base is not None:
            nn.init.trunc_normal_(self.hc_ffn_base, mean=0.0, std=0.02)
        if self.hc_ffn_scale is not None:
            nn.init.trunc_normal_(self.hc_ffn_scale, mean=0.0, std=0.02)
        if self.hc_attn_fn is not None:
            nn.init.trunc_normal_(self.hc_attn_fn, mean=0.0, std=0.02)
        if self.hc_attn_base is not None:
            nn.init.trunc_normal_(self.hc_attn_base, mean=0.0, std=0.02)
        if self.hc_attn_scale is not None:
            nn.init.trunc_normal_(self.hc_attn_scale, mean=0.0, std=0.02)


class HcHead(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm_eps: float
        hc_eps: float

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_eps = config.norm_eps
        self.hc_eps = config.hc_eps

    def forward(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)


class MTPModule(DeepSeekV4TransformerBlock):
    """
    MTP block with linear projection and transformerblock layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_id: int
        model_args: "DeepSeekV4Model.Config"

    def __init__(self, config: Config):
        super().__init__(
            DeepSeekV4TransformerBlock.Config(
                layer_id=config.layer_id, model_args=config.model_args
            )
        )
        model_args = config.model_args
        self.enorm = RMSNorm.Config(dim=model_args.dim, eps=model_args.norm_eps).build()
        self.hnorm = RMSNorm.Config(dim=model_args.dim, eps=model_args.norm_eps).build()
        self.e_proj = Linear.Config(
            in_features=model_args.dim,
            out_features=model_args.dim,
            bias=False,
        ).build()
        self.h_proj = Linear.Config(
            in_features=model_args.dim,
            out_features=model_args.dim,
            bias=False,
        ).build()
        self.hc_mult = hc_mult = model_args.hc_mult

    # pyrefly: ignore [bad-param-name-override]
    def forward(
        self,
        input_offset: torch.Tensor,
        prev_embed: torch.Tensor,
        input_ids: torch.Tensor,
        freqs_cis: torch.Tensor,
        hadamard_mat: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ):
        """
        Forward pass for the Transformer block.
        Args:
            input_offset (torch.Tensor): Input tensor of original token (batch_size, seq_len, dim).
            prev_embed (torch.Tensor): Input tensor of main module output token (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        input_offset = self.enorm(input_offset)
        prev_embed = self.hnorm(prev_embed)
        x = self.e_proj(input_offset) + self.h_proj(prev_embed)
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attention_norm(x)
        (
            x,
            compress_topk_idxs,
            offset,
            q,
            kv_compress,
            attention_masks,
            index_score,
            q_indexer,
            k_indexer,
            weights,
        ) = self.attention(x, freqs_cis, hadamard_mat, attention_masks)
        if self.attention.compress_ratio > 1 and hasattr(
            self.attention.pre_attention, "indexer"
        ):
            loss = self.cal_index_loss(
                compress_topk_idxs,
                offset,
                q,
                kv_compress,
                attention_masks,
                index_score,
                q_indexer,
                k_indexer,
                weights,
            )
            x = DSAIndexerLossAutoScaler.apply(x, loss)

        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.moe(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x

    def init_weights(self, buffer_device: torch.device):
        super().init_weights(buffer_device=buffer_device)
        for norm in (self.enorm, self.hnorm):
            nn.init.trunc_normal_(norm.weight, mean=1, std=0.02)
        nn.init.trunc_normal_(self.e_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.h_proj.weight, mean=0.0, std=0.02)


class DeepSeekV4Model(BaseModel):
    """
    DeepSeek-V4 Transformer model with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        norm_eps: float = 1e-6
        index_n_heads: int = 64
        index_head_dim: int = 128
        index_topk: int = 512
        enable_indexer_loss: bool = True
        save_format: str = "dcp"
        save_expert_format: str | None = None
        hf_save_dir: str | None = None
        dim: int = 4096
        moe_args: MoEArgs = field(default_factory=MoEArgs)
        rope_head_dim: int = 64
        q_lora_rank: int = 1024
        max_batch_size: int = 4
        max_seq_len: int = 4096
        n_heads: int = 64
        o_lora_rank: int = 1024
        head_dim: int = 512
        o_groups: int = 8
        window_size: int = 128
        compress_ratios: tuple[int, ...] = (1, 1, 4, 128)
        debug_force_load_balance: bool = False
        hc_sinkhorn_iters: int = 20
        hc_mult: int = 4
        hc_eps: float = 1e-6
        vocab_size: int = 129280
        moe_inter_dim: int = 2048
        load_balance_coeff: float = 1e-3
        compress_rope_theta: float = 40000.0
        original_seq_len: int = 65536
        rope_theta: int = 10000
        rope_factor: int = 4
        beta_fast: int = 32
        beta_slow: int = 1
        n_layers: int = 4
        use_sfa: bool = False
        num_mtp_modules: int = 0
        mtp_layer_compress_ratio: int = 1

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            seq_len = trainer_config.training.seq_len
            if seq_len > self.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
                )
            self.max_seq_len = seq_len

            self.moe_args.debug_force_load_balance = (
                trainer_config.debug.moe_force_load_balance
            )
            self.moe_args.load_balance_coeff = self.load_balance_coeff
            self.moe_args.n_hash_layers = 0
            self.use_sfa = False
            self.num_mtp_modules = getattr(trainer_config.training, "num_mtp_modules", 0)

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            # DeepSeek-V4 model config is not a `Decoder.Config`, so we cannot
            # use `get_moe_model_nparams_and_flops` (which expects `.layers[*].moe`).
            # Use the dense estimator over all parameters as a stable fallback.
            # [TODO] need to check
            return get_dense_model_nparams_and_flops(
                model=model,
                n_layers=self.n_layers + self.num_mtp_modules,
                n_heads=self.n_heads,
                head_dims=self.head_dim + self.head_dim,
                seq_len=seq_len,
            )

    def __init__(self, config: DeepSeekV4Model.Config):
        super().__init__()
        model_args = config
        self.max_seq_len = model_args.max_seq_len
        self.norm_eps = model_args.norm_eps
        self.layers = ModuleDict()
        for layer_id in range(model_args.n_layers + model_args.num_mtp_modules):
            if layer_id < model_args.n_layers:
                self.layers[str(layer_id)] = DeepSeekV4TransformerBlock.Config(
                    layer_id=layer_id, model_args=model_args
                ).build()
            else:
                self.layers[str(layer_id)] = MTPModule.Config(
                    layer_id=layer_id, model_args=model_args
                ).build()
        self.norm = RMSNorm.Config(dim=model_args.dim, eps=self.norm_eps).build()
        self.hc_eps = model_args.hc_eps
        self.hc_mult = hc_mult = model_args.hc_mult
        hc_dim = hc_mult * model_args.dim
        origin_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
        self.hc_head_scale = nn.Parameter(torch.empty(1))
        torch.set_default_dtype(origin_dtype)
        self.hc_head = HcHead.Config(norm_eps=self.norm_eps, hc_eps=self.hc_eps).build()
        self.model_args = model_args
        self.tok_embeddings = Embedding.Config(
            num_embeddings=model_args.vocab_size,
            embedding_dim=model_args.dim,
        ).build()
        self.output = Linear.Config(
            in_features=model_args.dim,
            out_features=model_args.vocab_size,
            bias=False,
        ).build()
        self.lm_head = self.output  # ChunkedCELoss requires lm_head reference
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(model_args, True), persistent=False
        )
        self.register_buffer(
            "freqs_cis_wo_compressor",
            precompute_freqs_cis(model_args, False),
            persistent=False,
        )
        self.register_buffer(
            "hadamard_mat",
            torch.empty(model_args.index_head_dim, model_args.index_head_dim),
            persistent=False,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        _skip_lm_head: bool = False,
    ):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            input_ids (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            _skip_lm_head: If True, return the hidden states before the output
                projection (used by ChunkedCELoss).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, seq_len, vocab_size),
            or hidden states if _skip_lm_head is True.
        """
        seq_len = tokens.shape[1]
        seq_len -= self.model_args.num_mtp_modules
        if self.tok_embeddings is not None:
            input_ids = tokens[:, :seq_len].detach().long()
            h = self.tok_embeddings(tokens[:, :seq_len])
            h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        else:
            input_ids = tokens[:, :seq_len].detach().long()
            h = tokens[:, :seq_len]
        # Main model calculate
        layer_id = 0
        for layer in self.layers.values():
            if layer_id < self.model_args.n_layers:
                h = layer(
                    h,
                    input_ids,
                    self.freqs_cis
                    # pyrefly: ignore [bad-index]
                    if self.model_args.compress_ratios[layer.layer_id] > 1
                    else self.freqs_cis_wo_compressor,
                    self.hadamard_mat,
                    attention_masks,
                )
            else:
                break
            layer_id += 1
        h = (
            self.hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
            if self.hc_head is not None
            else h
        )
        prev_embed = h
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h.float()) if self.output is not None else h
        if self.model_args.num_mtp_modules <= 0:
            if getattr(self, "_skip_lm_head", False):
                return h
            return output
        else:
            output_list = [None] * (1 + self.model_args.num_mtp_modules)
            # pyrefly: ignore [unsupported-operation]
            output_list[0] = output
            # MTP module calculate
            for mtp_layer_id in range(self.model_args.num_mtp_modules):
                token_offset_id = mtp_layer_id + 1
                token_end_idx = token_offset_id + seq_len
                token_offset = tokens[:, token_offset_id:token_end_idx]
                input_offset = self.tok_embeddings(token_offset)
                layer_id = mtp_layer_id + self.model_args.n_layers
                h = self.layers[str(layer_id)](
                    input_offset,
                    prev_embed,
                    input_ids,
                    # Assumption: No compressor in MTP modules.
                    self.freqs_cis_wo_compressor,
                    self.hadamard_mat,
                    attention_masks,
                )
                h = (
                    self.hc_head(
                        h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
                    )
                    if self.hc_head is not None
                    else h
                )
                prev_embed = h
                h = self.norm(h) if self.norm is not None else h
                output = self.output(h.float()) if self.output is not None else h
                output_list[mtp_layer_id + 1] = output
        return output_list

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(self.model_args, True)
            self.freqs_cis_wo_compressor = precompute_freqs_cis(self.model_args, False)
            import scipy  # lazy import — only needed for hadamard matrix
            self.hadamard_mat = torch.tensor(
                # pyrefly: ignore [implicit-import]
                scipy.linalg.hadamard(self.model_args.index_head_dim, float),
                dtype=torch.bfloat16,
            )
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                # pyrefly: ignore [not-callable]
                layer.init_weights(buffer_device=buffer_device)
        if self.norm is not None:
            nn.init.trunc_normal_(self.norm.weight, mean=1, std=0.02)
        if self.hc_head_fn is not None:
            nn.init.trunc_normal_(self.hc_head_fn, mean=0.0, std=0.02)
        if self.hc_head_base is not None:
            nn.init.trunc_normal_(self.hc_head_base, mean=0.0, std=0.02)
        if self.hc_head_scale is not None:
            nn.init.trunc_normal_(self.hc_head_scale, mean=0.0, std=0.02)
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )
