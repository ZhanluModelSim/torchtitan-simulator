# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# This file is copied from torchtitan,
# https://github.com/pytorch/torchtitan/blob/v0.2.2/torchtitan/models/deepseek_v3/model/state_dict_adapter.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import logging
import re
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.tensor import DTensor
from torchtitan.models.deepseek_v3 import DeepSeekV3StateDictAdapter
from torchtitan.models.utils import MoEStateDictAdapter


logger = logging.getLogger(__name__)


class DeepSeekV4StateDictAdapter(DeepSeekV3StateDictAdapter):
    def __init__(self, model_args, hf_assets_path: str | None = None):
        MoEStateDictAdapter.__init__(self, model_args, hf_assets_path)

        # key mapping
        self.from_hf_map = {
            "embed.weight": "tok_embeddings.weight",
            # Attention Module
            "layers.{}.attn.attn_sink": "layers.{}.attention.inner_attention.attn_sink",
            "layers.{}.attn.kv_norm.weight": "layers.{}.attention.pre_attention.kv_norm.weight",
            "layers.{}.attn.q_norm.weight": "layers.{}.attention.pre_attention.q_norm.weight",
            "layers.{}.attn.wo_a.weight": "layers.{}.attention.post_attention.wo_a.weight",
            "layers.{}.attn.wkv.weight": "layers.{}.attention.pre_attention.wkv.weight",
            "layers.{}.attn.wo_b.weight": "layers.{}.attention.post_attention.wo_b.weight",
            "layers.{}.attn.wq_a.weight": "layers.{}.attention.pre_attention.wq_a.weight",
            "layers.{}.attn.wq_b.weight": "layers.{}.attention.pre_attention.wq_b.weight",
            # Transformer Layer
            "layers.{}.attn_norm.weight": "layers.{}.attention_norm.weight",
            "layers.{}.ffn_norm.weight": "layers.{}.ffn_norm.weight",
            # MoE Module
            "layers.{}.ffn.experts.{}.w1.weight": "layers.{}.moe.experts.w1",
            "layers.{}.ffn.experts.{}.w3.weight": "layers.{}.moe.experts.w3",
            "layers.{}.ffn.experts.{}.w2.weight": "layers.{}.moe.experts.w2",
            "layers.{}.ffn.gate.weight": "layers.{}.moe.router.gate.weight",
            "layers.{}.ffn.gate.bias": "layers.{}.moe.expert_bias",
            "layers.{}.ffn.shared_experts.w1.weight": "layers.{}.moe.shared_experts.w1.weight",
            "layers.{}.ffn.shared_experts.w3.weight": "layers.{}.moe.shared_experts.w3.weight",
            "layers.{}.ffn.shared_experts.w2.weight": "layers.{}.moe.shared_experts.w2.weight",
            # mHC
            "layers.{}.hc_attn_base": "layers.{}.hc_attn_base",
            "layers.{}.hc_attn_fn": "layers.{}.hc_attn_fn",
            "layers.{}.hc_attn_scale": "layers.{}.hc_attn_scale",
            "layers.{}.hc_ffn_base": "layers.{}.hc_ffn_base",
            "layers.{}.hc_ffn_fn": "layers.{}.hc_ffn_fn",
            "layers.{}.hc_ffn_scale": "layers.{}.hc_ffn_scale",
            "hc_head_base": "hc_head_base",
            "hc_head_fn": "hc_head_fn",
            "hc_head_scale": "hc_head_scale",
            "norm.weight": "norm.weight",
            "head.weight": "output.weight",
        }
        self.compress_ratios = model_args.compress_ratios
        for layer_id in range(model_args.n_layers):
            cr = self.compress_ratios[layer_id]
            # compressor
            if cr != 1:
                compressor_attr = "compressor" if cr == 4 else "compressor_128"
                self.from_hf_map.update(
                    {
                        f"layers.{layer_id}.attn.compressor.ape": (
                            f"layers.{layer_id}.attention.pre_attention.{compressor_attr}.ape"
                        ),
                        f"layers.{layer_id}.attn.compressor.norm.weight": (
                            f"layers.{layer_id}.attention.pre_attention.{compressor_attr}.norm.weight"
                        ),
                        f"layers.{layer_id}.attn.compressor.wgate.weight": (
                            f"layers.{layer_id}.attention.pre_attention.{compressor_attr}.wgate.weight"
                        ),
                        f"layers.{layer_id}.attn.compressor.wkv.weight": (
                            f"layers.{layer_id}.attention.pre_attention.{compressor_attr}.wkv.weight"
                        ),
                    }
                )
            # indexer
            if cr == 4:
                self.from_hf_map.update(
                    {
                        f"layers.{layer_id}.attn.indexer.compressor.ape": (
                            f"layers.{layer_id}.attention.pre_attention.indexer.compressor.ape"
                        ),
                        f"layers.{layer_id}.attn.indexer.compressor.norm.weight": (
                            f"layers.{layer_id}.attention.pre_attention.indexer.compressor.norm.weight"
                        ),
                        f"layers.{layer_id}.attn.indexer.compressor.wgate.weight": (
                            f"layers.{layer_id}.attention.pre_attention.indexer.compressor.wgate.weight"
                        ),
                        f"layers.{layer_id}.attn.indexer.compressor.wkv.weight": (
                            f"layers.{layer_id}.attention.pre_attention.indexer.compressor.wkv.weight"
                        ),
                        f"layers.{layer_id}.attn.indexer.wq_b.weight": (
                            f"layers.{layer_id}.attention.pre_attention.indexer.wq_b.weight"
                        ),
                        f"layers.{layer_id}.attn.indexer.weights_proj.weight": (
                            f"layers.{layer_id}.attention.pre_attention.indexer.weights_proj.weight"
                        ),
                    }
                )
            if layer_id <= model_args.moe_args.n_hash_layers - 1:
                self.from_hf_map.update(
                    {
                        f"layers.{layer_id}.ffn.gate.tid2eid": f"layers.{layer_id}.moe.router.tid2eid",
                    }
                )
        if model_args.num_mtp_modules > 0:
            for mtp_layer_id in range(model_args.num_mtp_modules):
                layer_id = mtp_layer_id + model_args.n_layers
                self.from_hf_map.update(
                    {
                        "layers.{}.enorm.weight": "layers.{}.enorm.weight",
                        "layers.{}.hnorm.weight": "layers.{}.hnorm.weight",
                        "layers.{}.e_proj.weight": "layers.{}.e_proj.weight",
                        "layers.{}.h_proj.weight": "layers.{}.h_proj.weight",
                    }
                )
        # configs
        self.use_gmm = getattr(model_args.moe_args, "use_grouped_mm", False)
        self.n_experts = model_args.moe_args.num_experts
        self.first_k_dense = getattr(model_args.moe_args, "first_k_dense", 0)
        self._input_format = "hf"
        self._input_expert_format = "standard"
        self._to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        self._to_hf_passthrough = (
            "tid2eid",
            "compressor",
            "indexer",
        )
        self._from_hf_passthrough = (
            "tid2eid",
            "compressor",
            "indexer",
        )

    @staticmethod
    def _get_abstract_key(key: str, count: int) -> str:
        return re.sub(r"(\d+)", "{}", key, count=count)

    @staticmethod
    def _first_number(key: str) -> str:
        # pyrefly: ignore [missing-attribute]
        return re.search(r"\d+", key).group(0)

    @classmethod
    def _map_layer_key(cls, key: str, mapping: dict[str, str]) -> str:
        abstract_key = cls._get_abstract_key(key, count=1)
        layer_num = cls._first_number(key)
        return mapping[abstract_key].format(layer_num)

    def to_hf_mtp(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        new_state_dict = {}
        base = self.model_config.n_layers
        mtp_count = self.model_config.num_mtp_modules
        for key, tensor in state_dict.items():
            match = re.match(r"layers\.(\d+)\.(.+)", key)
            if not match:
                new_state_dict[key] = tensor
                continue

            layer_idx = int(match.group(1))
            suffix = match.group(2)
            mtp_idx = layer_idx - base
            if 0 <= mtp_idx < mtp_count:
                new_state_dict[f"mtp.{mtp_idx}.{suffix}"] = tensor
            else:
                new_state_dict[key] = tensor
        return new_state_dict

    def to_hf_deepseekv4(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. Convert between the HF shape and the torchtitan shape.
        2. Split the GroupedExperts' weight into separate expert's weight.
        """
        to_hf_map = self._to_hf_map
        passthrough = self._to_hf_passthrough
        hf_state_dict = {}
        for key, value in state_dict.items():
            if any(token in key for token in passthrough):
                new_key = to_hf_map[key]
                # uses torch.finfo() which only supports float types. Convert to float
                # here and convert back in from_hf_deepseekv4 on load.
                if "tid2eid" in key:
                    value = value.to(torch.float32)
                hf_state_dict[new_key] = value
            elif "moe.experts" in key:
                abstract_key = self._get_abstract_key(key, count=1)
                layer_num = self._first_number(key)
                new_abstract_key = to_hf_map[abstract_key]

                # Store the GroupedExperts Weight metadata for from_hf()
                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[
                        abstract_key
                    ] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[abstract_key] = value.device_mesh

                    # Split GroupedExperts weight to local individual expert weights
                    local_expert_fqn = self._get_local_experts_weights(
                        new_abstract_key,
                        abstract_key,
                        layer_num,
                        value,
                    )
                    hf_state_dict.update(local_expert_fqn)

                else:
                    # keep this path for offline conversion
                    split_values = self._split_experts_weights(
                        value,
                        self.n_experts,
                    )

                    for expert_num in range(0, self.n_experts):
                        new_key = new_abstract_key.format(layer_num, expert_num)
                        hf_state_dict[new_key] = split_values[expert_num].squeeze()

            elif "layers" in key:
                hf_state_dict[self._map_layer_key(key, to_hf_map)] = value

            else:
                new_key = to_hf_map[key]
                hf_state_dict[new_key] = value

        hf_state_dict = self.to_hf_mtp(hf_state_dict)

        return hf_state_dict

    # pyrefly: ignore [bad-override]
    def get_hf_storage_reader(self, path: str, from_quantized: bool = False):
        if path.endswith(".safetensors"):
            self._input_format = "hf"
            return HuggingFaceStorageReader(path)
        else:
            self._input_format = "dcp"
            from torch.distributed.checkpoint import FileSystemReader

            return FileSystemReader(path)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Create a load plan/ Convert to HF format"""
        return self.to_hf_deepseekv4(state_dict)

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert loaded data to runtime format"""
        filtered = {
            k: v
            for k, v in hf_state_dict.items()
            if not k.endswith(".weight_scale_inv")
        }

        if self._input_format == "hf":
            state_dict = self.from_hf_deepseekv4(filtered)
        else:
            state_dict = filtered

        return state_dict

    def from_hf_mtp(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        new_state_dict = {}
        base = self.model_config.n_layers
        mtp_count = self.model_config.num_mtp_modules
        for key, tensor in state_dict.items():
            match = re.match(r"mtp\.(\d+)\.(.+)", key)
            if not match:
                new_state_dict[key] = tensor
                continue

            mtp_idx = int(match.group(1))
            suffix = match.group(2)
            if 0 <= mtp_idx < mtp_count:
                new_state_dict[f"layers.{base + mtp_idx}.{suffix}"] = tensor
            else:
                new_state_dict[key] = tensor
        return new_state_dict

    def from_hf_deepseekv4(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. When loading from HF checkpoint, dequantize the weights from float8 to float32.
        2. Convert between the HF shape and the torchtitan shape.
        3. Concat separate expert's weight into GroupedExperts' weight.
        """

        state_dict = {}
        expert_weights_by_layer = {}  # {layer: {abstract_key: {expert_id: tensor}}}
        passthrough = self._from_hf_passthrough

        hf_state_dict = self.from_hf_mtp(hf_state_dict)

        for key, value in hf_state_dict.items():
            if any(token in key for token in passthrough):
                new_key = self.from_hf_map[key]
                # tid2eid was saved as float32 to work around torch.finfo() issue
                # in safetensors consolidation. Convert back to int64.
                if "tid2eid" in key:
                    value = value.to(torch.int64)
                state_dict[new_key] = value
            elif "ffn.experts" in key:
                abstract_key = self._get_abstract_key(key, count=2)
                layer_num, expert_num, _ = re.findall(r"\d+", key)
                titan_abstract_key = self.from_hf_map[abstract_key]
                new_key = titan_abstract_key.format(layer_num)

                # Store the expert's weight in expert_weights_by_layer for concatenating later.
                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}
                if titan_abstract_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][titan_abstract_key] = {}
                expert_weights_by_layer[layer_num][titan_abstract_key][
                    int(expert_num)
                ] = value

                # Use stored metadata to decide path (online vs offline)
                # Online mode: local_experts_indices was populated during to_hf()
                if titan_abstract_key in self.local_experts_indices:
                    stacked_value = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                    )
                else:  # keep this path to be compatible with offline conversion
                    stacked_value = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        # pyrefly: ignore [missing-attribute]
                        self.model_config.moe_args.num_experts,
                    )

                if stacked_value is not None:
                    state_dict[new_key] = stacked_value

            elif "layers" in key:
                state_dict[self._map_layer_key(key, self.from_hf_map)] = value

            else:
                new_key = self.from_hf_map[key]
                state_dict[new_key] = value

        return state_dict
