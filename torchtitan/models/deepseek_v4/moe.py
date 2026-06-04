# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.common import moe as common_moe
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module

GroupedExperts = common_moe.GroupedExperts


class TokenReorderer(Module):
    """Minimal CPU-compatible token reorderer (NPU patching removed).

    In the original NPU fork this was patched by ``npu_gmm`` to use
    Ascend-only grouped-matrix-multiply kernels.  This stub returns
    the routing metadata unchanged so that the model works on CPU.
    """

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Group tokens by expert: sort by expert index so that tokens
        # for the same expert are contiguous.  Divide by top_k because
        # the downstream code indexes into x_flat (shape [bs*seqlen, dim])
        # while expert indices are (bs, seqlen, top_k).
        _, token_indices_experts_sorted = torch.sort(
            selected_experts_indices.reshape(-1), stable=True
        )
        token_indices_experts_sorted = (
            token_indices_experts_sorted // self.top_k
        )
        token_counts = torch.bincount(
            selected_experts_indices.flatten().long(),
            minlength=self.num_experts,
        )
        return top_scores, token_indices_experts_sorted, token_counts


@dataclass
class MoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1

    # router
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_norm: bool = False
    route_scale: float = 1.5
    gate_bias: bool = False
    score_before_experts: bool = False

    # token-choice with optional node limited routing
    top_k: int = 1
    num_expert_groups: int | None = None  # if set, must divide num_experts
    num_limited_groups: int | None = 8
    use_grouped_mm: bool = True  # grouped mm or for-loop for the experts computation
    load_balance_coeff: float | None = 1e-3

    debug_force_load_balance: bool = False
    # if True, we force each experts get same amount of token via round-robin

    n_hash_layers: int = 3
    swiglu_limit: float = 10


def _softplus_stable(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.exp(-x.abs())) + torch.relu(x)


def _build_hash_routing_table(
    vocab_size: int,
    num_experts: int,
    top_k: int,
    *,
    device: torch.device | str | None = None,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Build ``tid2eid`` with top_k distinct expert ids for each token id."""
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")

    tid2eid = torch.empty((vocab_size, top_k), dtype=torch.long, device=device)
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        tid2eid[start:end] = (
            torch.rand((end - start, num_experts), device=device)
            .topk(top_k, dim=-1)
            .indices
        )
    return tid2eid


class TokenChoiceTopKRouter(common_moe.TokenChoiceTopKRouter):
    """DSV4 router: sqrtsoftplus + optional hash routing; gate is nn.Linear."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_experts: int
        top_k: int
        layer_id: int
        args: "MoEArgs"
        score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"]
        route_norm: bool
        route_scale: float
        vocab_size: int
        debug_force_load_balance: bool = False

    def __init__(self, config: Config):
        # Do not call common_moe.TokenChoiceTopKRouter.__init__: upstream expects
        # Linear.Config gate and only sigmoid/softmax score_func.
        super(common_moe.TokenChoiceTopKRouter, self).__init__()
        dim = config.dim
        num_experts = config.num_experts
        top_k = config.top_k
        layer_id = config.layer_id
        args = config.args
        score_func = config.score_func
        route_norm = config.route_norm
        route_scale = config.route_scale
        vocab_size = config.vocab_size
        debug_force_load_balance = config.debug_force_load_balance

        self.gate = Linear.Config(
            in_features=dim,
            out_features=num_experts,
            bias=False,
        ).build()
        self.num_expert_groups = args.num_expert_groups
        self.num_limited_groups = args.num_limited_groups
        self.num_experts = num_experts
        self.top_k = top_k
        # pyrefly: ignore [bad-assignment]
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self._debug_force_load_balance = debug_force_load_balance
        self.hash = layer_id < args.n_hash_layers
        self.vocab_size = vocab_size
        if self.hash:
            self.register_buffer(
                "tid2eid",
                _build_hash_routing_table(
                    self.vocab_size, self.num_experts, self.top_k
                ),
                persistent=True,
            )

    # pyrefly: ignore [bad-override]
    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        expert_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        scores = self.gate(x)
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        elif self.score_func == "sqrtsoftplus":
            scores = _softplus_stable(scores.to(torch.float32)).sqrt()
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if self.hash:
            selected_experts_indices = self.tid2eid[input_ids.flatten()]
        else:
            # expert_bias is used only in non-hash layers.
            scores_for_choice = scores if expert_bias is None else scores + expert_bias
            selected_experts_indices = scores_for_choice.topk(self.top_k, dim=-1)[1]

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        top_scores = scores.gather(dim=1, index=selected_experts_indices)
        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
            ) = self._debug_force_load_balance_routing(scores)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.bincount(
            selected_experts_indices.view(-1).long(),
            minlength=self.num_experts,
        )
        return top_scores, selected_experts_indices, num_tokens_per_expert

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)
        if self.hash:
            self.tid2eid = _build_hash_routing_table(
                self.vocab_size,
                self.num_experts,
                self.top_k,
                device=self.gate.weight.device,
            )


class MoE(Module):
    """DSV4 token-choice MoE (hash routing optional). Mirrors upstream MoE wiring."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        moe_args: MoEArgs
        dim: int
        hidden_dim: int
        layer_id: int
        vocab_size: int

    def __init__(self, config: Config):
        super().__init__()
        moe_args = config.moe_args
        dim = config.dim
        hidden_dim = config.hidden_dim
        layer_id = config.layer_id
        vocab_size = config.vocab_size
        num_experts = moe_args.num_experts

        self.experts = GroupedExperts.Config(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            token_dispatcher=common_moe.LocalTokenDispatcher.Config(
                num_experts=num_experts,
                top_k=moe_args.top_k,
            ),

        ).build()


        self.router = TokenChoiceTopKRouter.Config(
            dim=dim,
            layer_id=layer_id,
            args=moe_args,
            num_experts=num_experts,
            top_k=moe_args.top_k,
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
            vocab_size=vocab_size,
            debug_force_load_balance=moe_args.debug_force_load_balance,
        ).build()

        self.reorderer = TokenReorderer(
            num_experts=num_experts,
            top_k=moe_args.top_k,
        )

        n_shared = moe_args.num_shared_experts
        if n_shared > 0:
            shared_hidden = hidden_dim * n_shared
            self.shared_experts = FeedForward.Config(
                w1=Linear.Config(
                    in_features=dim,
                    out_features=shared_hidden,
                    bias=False,
                ),
                w2=Linear.Config(
                    in_features=shared_hidden,
                    out_features=dim,
                    bias=False,
                ),
                w3=Linear.Config(
                    in_features=dim,
                    out_features=shared_hidden,
                    bias=False,
                ),
            ).build()
        else:
            self.shared_experts = None

        self.score_before_experts = moe_args.score_before_experts
        self.load_balance_coeff = moe_args.load_balance_coeff

        if self.load_balance_coeff is not None:
            if self.load_balance_coeff <= 0.0:
                raise ValueError("load_balance_coeff must be greater than 0.0")
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None

        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

        # Remove expert_bias buffer for hash layers. Note that init_weight of
        # class MoE will still create a non-buffer field named as `expert_bias`
        # so that `build_optimizers_with_moe_load_balancing` will not break.
        if layer_id < moe_args.n_hash_layers:
            del self.expert_bias

    def init_weights(self, init_std: float, buffer_device: torch.device | None = None):
        self.router.init_weights(init_std)
        experts_init = getattr(self.experts, "init_weights", None)
        if callable(experts_init):
            experts_init(init_std)
        else:
            for name in ("w1", "w2", "w3"):
                p = getattr(self.experts, name, None)
                if p is not None:
                    nn.init.trunc_normal_(p, mean=0.0, std=init_std)
        if self.shared_experts is not None:
            for m in self.shared_experts.modules():
                if isinstance(m, nn.Linear) and m.weight is not None:
                    nn.init.trunc_normal_(m.weight, mean=0.0, std=init_std)

    # pyrefly: ignore [bad-override]
    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DeepSeek MoE module.

        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.
            input_ids (torch.Tensor): Token IDs tensor for hash-based routing.

        Returns:
            torch.Tensor: Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape
        x_flat = x.view(-1, dim)
        input_ids_flat = input_ids.flatten()
        bias = getattr(self, "expert_bias", None)
        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x_flat,
            input_ids_flat,
            bias,
        )

        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        # Use torchtitan's GroupedExperts API, which handles dispatch,
        # expert computation, score weighting, and combine internally.
        out_experts = self.experts(
            x_flat,
            top_scores,
            selected_experts_indices,
            shared_experts=self.shared_experts,
        )

        output = out_experts.reshape(bs, slen, dim)

        return output
