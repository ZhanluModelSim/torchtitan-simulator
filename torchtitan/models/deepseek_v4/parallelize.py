# Adapted from
# https://github.com/pytorch/torchtitan/blob/v0.2.1/torchtitan/models/deepseek_v3/infra/parallelize.py
# https://github.com/pytorch/torchtitan/blob/v0.2.1/torchtitan/models/llama4/infra/parallelize.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
)
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.expert_parallel import (
    ExpertParallel,
    TensorParallel,
)
try:
    from torchtitan.distributed.expert_parallel import DeepEPExpertParallel
except ImportError:
    DeepEPExpertParallel = None  # type: ignore[assignment]
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp, NoParallel
from torchtitan.models.common import moe as moe_module
from torchtitan.models.llama3.parallelize import apply_fsdp as _apply_fsdp_upstream

# apply_replicate doesn't exist in llama3.parallelize in this torchtitan version.
# Provide a minimal stub.
def apply_replicate(model: nn.Module, dp_mesh: DeviceMesh):
    return model

def apply_fsdp(*args, **kwargs):
    return _apply_fsdp_upstream(*args, **kwargs)

from .model import (
    Attention,
    DSAIndexerLossLoggingHelper,
)


logger = logging.getLogger(__name__)


# for selective op activation checkpointing
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops._c10d_functional.all_to_all_single.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
    torch._higher_order_ops.inductor_compiled_code,
}


class AwaitRowwiseParallel(RowwiseParallel):
    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # Rowwise sharding produces partial output, depending on output layouts:
        # 1. to replicate -> allreduce
        # 2. to shard -> reduce_scatter
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)

        # wait for async redistribution to complete
        real_tensor = outputs._local_tensor
        torch.ops._c10d_functional.wait_tensor(real_tensor)

        # back to local tensor if use_local_output is True
        return outputs.to_local() if use_local_output else outputs


def _register_distributed_parameter(
    module: nn.Module,
    name: str,
    device_mesh: DeviceMesh,
    placements: list,
):
    dt = nn.Parameter(
        distribute_tensor(
            getattr(module, name),
            device_mesh=device_mesh,
            placements=placements,
            src_data_rank=0,
        )
    )
    module.register_parameter(name, dt)


def parallelize_deepseek_v4(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training,
    model_converters,
    parallelism,
    compile_config,
    ac_config,
    dump_folder: str,
):
    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    # patch the indexer loss tracking with distributed version to get the synchronized indexer loss metric
    apply_distributed_indexer_loss_tracking(parallel_dims)

    if parallel_dims.tp_enabled:
        tp_mesh = parallel_dims.get_mesh("tp")
        apply_non_moe_tp(
            model,
            tp_mesh,
            loss_parallel=not parallelism.disable_loss_parallel,
            parallelism=parallelism,
            ac_config=ac_config,
        )
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    # Check if using DeepEP for MoE communication
    if parallelism.expert_parallel_comm_backend == "deepep" and DeepEPExpertParallel is not None:
        if not parallel_dims.ep_enabled:
            raise ValueError(
                "DeepEP requires expert parallelism (ep_degree > 1). "
                "The DeepEP MoE model code does not support EP=1. "
                "Please set expert_parallel_degree > 1 or use standard communication backend."
            )
        if parallel_dims.etp_enabled:
            raise NotImplementedError(
                "DeepEP with Expert Tensor Parallelism (ETP) is not supported yet. "
                "Please set expert_tensor_parallel_degree=1 or use standard communication backend."
            )

        use_deepep = True

        # Import deepep module to register custom ops before accessing them
        import torchtitan.distributed.deepep  # noqa: F401 - registers torch.ops.deepep

        _op_sac_save_list.add(torch.ops.deepep.dispatch.default)
        _op_sac_save_list.add(torch.ops.deepep.combine.default)
    else:
        use_deepep = False

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
            use_deepep=use_deepep,
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    if model_compile_enabled:
        apply_compile(model, compile_config, parallel_dims.ep_enabled)

    dp_mesh: DeviceMesh | None = None
    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        # apply FSDP or HSDP, potentially with Context Parallel
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")

    elif parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh("dp_replicate")
        if dp_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_replicate(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    parallelism,
    ac_config,
):
    """Apply tensor parallelism."""

    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    (
        rowwise_parallel,
        await_rowwise_parallel,
        colwise_parallel,
        prepare_module_input,
        prepare_module_input_output,
    ) = (
        RowwiseParallel,
        AwaitRowwiseParallel,
        ColwiseParallel,
        PrepareModuleInput,
        PrepareModuleInputOutput,
    )
    hc_head_plan = prepare_module_input_output(
        input_layouts=(Shard(1), Replicate(), Replicate(), Replicate()),
        desired_input_layouts=(Replicate(), Replicate(), Replicate(), Replicate()),
        use_local_input=True,
        output_layouts=(Replicate()),
        desired_output_layouts=(Shard(1)),
        use_local_output=False,
    )
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
            "hc_head": hc_head_plan,
        },
    )
    _register_distributed_parameter(model, "hc_head_fn", tp_mesh, [Replicate()])
    _register_distributed_parameter(model, "hc_head_base", tp_mesh, [Replicate()])
    _register_distributed_parameter(model, "hc_head_scale", tp_mesh, [Replicate()])

    attention_kernel_plan_ratio1 = prepare_module_input_output(
        input_layouts=(Shard(2), Replicate(), Shard(0), None, None),
        desired_input_layouts=(Shard(2), Replicate(), Shard(0), None, None),
        use_local_input=True,
        output_layouts=(Shard(2)),
        desired_output_layouts=(Shard(2)),
        use_local_output=False,
    )

    attention_kernel_plan_ratio4 = prepare_module_input_output(
        input_layouts=(Shard(2), Replicate(), Shard(0), Replicate(), Replicate()),
        desired_input_layouts=(
            Shard(2),
            Replicate(),
            Shard(0),
            Replicate(),
            Replicate(),
        ),
        use_local_input=True,
        output_layouts=(Shard(2)),
        desired_output_layouts=(Shard(2)),
        use_local_output=False,
    )

    attention_kernel_plan_ratio128 = prepare_module_input_output(
        input_layouts=(Shard(2), Replicate(), Shard(0), Replicate(), None),
        desired_input_layouts=(Shard(2), Replicate(), Shard(0), Replicate(), None),
        use_local_input=True,
        output_layouts=(Shard(2)),
        desired_output_layouts=(Shard(2)),
        use_local_output=False,
    )

    indexer_plan = prepare_module_input_output(
        input_layouts=(Replicate(), Replicate(), Replicate(), Replicate(), None),
        desired_input_layouts=(
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            None,
        ),
        use_local_input=True,
        output_layouts=(Replicate(), Replicate(), Replicate()),
        desired_output_layouts=(Replicate(), Replicate(), Replicate()),
        use_local_output=False,
    )

    li_compute_plan = prepare_module_input_output(
        input_layouts=(Replicate(), Replicate(), Replicate(), None, None),
        desired_input_layouts=(Replicate(), Replicate(), Replicate(), None, None),
        use_local_input=True,
        output_layouts=(Replicate(), Replicate()),
        desired_output_layouts=(Replicate(), Replicate()),
        use_local_output=False,
    )

    compressor_plan = prepare_module_input_output(
        input_layouts=(Replicate(), Replicate()),
        desired_input_layouts=(Replicate(), Replicate()),
        use_local_input=False,
        output_layouts=(Replicate()),
        desired_output_layouts=(Replicate()),
        use_local_output=False,
    )

    indexer_compressor_plan = prepare_module_input_output(
        input_layouts=(Replicate(), Replicate()),
        desired_input_layouts=(Replicate(), Replicate()),
        use_local_input=False,
        output_layouts=(Replicate()),
        desired_output_layouts=(Replicate()),
        use_local_output=True,
    )

    hc_pre_plan = prepare_module_input_output(
        input_layouts=(Shard(1), Replicate(), Replicate(), Replicate()),
        desired_input_layouts=(Replicate(), Replicate(), Replicate(), Replicate()),
        use_local_input=True,
        output_layouts=(Replicate(), Replicate(), Replicate()),
        desired_output_layouts=(Shard(1), Shard(1), Shard(1)),
        use_local_output=False,
    )

    hc_post_plan = prepare_module_input_output(
        input_layouts=(Shard(1), Shard(1), Shard(1), Shard(1)),
        desired_input_layouts=(Replicate(), Replicate(), Replicate(), Replicate()),
        use_local_input=True,
        output_layouts=(Replicate()),
        desired_output_layouts=(Shard(1)),
    )

    hc_pre_sinkhon_plan = prepare_module_input(
        input_layouts=(Replicate(), Replicate(), Replicate(), None, None, None),
        desired_input_layouts=(Replicate(), Replicate(), Replicate(), None, None, None),
        use_local_output=True,
    )

    get_window_topk_idxs_plan = prepare_module_input_output(
        input_layouts=(None, None, None),
        desired_input_layouts=(None, None, None),
        use_local_input=False,
        output_layouts=(Replicate()),
        desired_output_layouts=(Replicate()),
        use_local_output=True,
    )

    get_compress_topk_idxs_plan = prepare_module_input_output(
        input_layouts=(Replicate(), None),
        desired_input_layouts=(Replicate(), None),
        use_local_input=False,
        output_layouts=(Replicate()),
        desired_output_layouts=(Replicate()),
        use_local_output=True,
    )

    li_loss_plan = prepare_module_input(
        input_layouts=(
            Shard(1),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            None,
            None,
        ),
        desired_input_layouts=(
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            None,
            None,
        ),
        use_local_output=True,
    )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        _register_distributed_parameter(
            # pyrefly: ignore [missing-attribute]
            transformer_block.attention.inner_attention,
            "attn_sink",
            tp_mesh,
            [Shard(0)],
        )
        _register_distributed_parameter(
            # pyrefly: ignore [bad-argument-type]
            transformer_block,
            "hc_attn_fn",
            tp_mesh,
            [Replicate()],
        )
        _register_distributed_parameter(
            # pyrefly: ignore [bad-argument-type]
            transformer_block,
            "hc_ffn_fn",
            tp_mesh,
            [Replicate()],
        )
        _register_distributed_parameter(
            # pyrefly: ignore [bad-argument-type]
            transformer_block,
            "hc_attn_base",
            tp_mesh,
            [Replicate()],
        )
        _register_distributed_parameter(
            # pyrefly: ignore [bad-argument-type]
            transformer_block,
            "hc_ffn_base",
            tp_mesh,
            [Replicate()],
        )
        _register_distributed_parameter(
            # pyrefly: ignore [bad-argument-type]
            transformer_block,
            "hc_attn_scale",
            tp_mesh,
            [Replicate()],
        )
        _register_distributed_parameter(
            # pyrefly: ignore [bad-argument-type]
            transformer_block,
            "hc_ffn_scale",
            tp_mesh,
            [Replicate()],
        )

        # pyrefly: ignore [missing-attribute]
        if transformer_block.attention.compress_ratio == 1:
            attention_kernel_plan = attention_kernel_plan_ratio1
        # pyrefly: ignore [missing-attribute]
        elif transformer_block.attention.compress_ratio == 4:
            attention_kernel_plan = attention_kernel_plan_ratio4
        else:
            attention_kernel_plan = attention_kernel_plan_ratio128

        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), Replicate(), None, None),
                desired_input_layouts=(Replicate(), Replicate(), None, None),
            ),
            "attention.inner_attention.sparse_attn.get_window_topk_idxs": get_window_topk_idxs_plan,
            "attention.inner_attention.sparse_attn.get_compress_topk_idxs": get_compress_topk_idxs_plan,
            # NOTE: use_local_output=False make the output to be a DTensor instead of a plain Tensor
            # so that the intermedidate results k is generated as a DTensor and its gradient is
            # correctly handled by the autograd engine.
            "attention.pre_attention.wq_a": NoParallel(),
            "attention.pre_attention.q_norm": NoParallel(),
            "attention.pre_attention.wq_b": colwise_parallel(use_local_output=False),
            "attention.pre_attention.wkv": NoParallel(),
            "attention.pre_attention.kv_norm": NoParallel(),
            "attention.post_attention.wo_a": colwise_parallel(use_local_output=False),
            "attention.post_attention.wo_b": rowwise_parallel(
                input_layouts=Shard(-1),
                output_layouts=Shard(1),
                use_local_output=False,
            ),
            "attention.inner_attention.sparse_attn": attention_kernel_plan,
            "hc_post": hc_post_plan,
            "hc_pre": hc_pre_plan,
            "hc_pre.torch_hc_split_sinkhorn": hc_pre_sinkhon_plan,
            "cal_index_loss.li_loss": li_loss_plan,
            "ffn_norm": SequenceParallel(),
        }
        # pyrefly: ignore [missing-attribute]
        if transformer_block.attention.compress_ratio > 1:
            # pyrefly: ignore [missing-attribute]
            compress_ratio = transformer_block.attention.compress_ratio
            if compress_ratio == 4:
                compressor_attr = "compressor"
            else:
                compressor_attr = "compressor_128"
            compressor_module = getattr(
                # pyrefly: ignore [missing-attribute]
                transformer_block.attention.pre_attention,
                compressor_attr,
            )
            compressor_key = f"attention.pre_attention.{compressor_attr}"
            _register_distributed_parameter(
                compressor_module, "ape", tp_mesh, [Replicate()]
            )
            layer_plan.update(
                {
                    compressor_key: compressor_plan,
                    f"{compressor_key}.wkv": NoParallel(),
                    f"{compressor_key}.wgate": NoParallel(),
                    f"{compressor_key}.norm": NoParallel(),
                }
            )
            if compress_ratio == 4:
                _register_distributed_parameter(
                    # pyrefly: ignore [missing-attribute]
                    transformer_block.attention.pre_attention.indexer.compressor,
                    "ape",
                    tp_mesh,
                    [Replicate()],
                )
                layer_plan.update(
                    {
                        "attention.inner_attention.li_compute": li_compute_plan,
                        "attention.pre_attention.indexer": indexer_plan,
                        "attention.pre_attention.indexer.compressor": indexer_compressor_plan,
                        "attention.pre_attention.indexer.wq_b": NoParallel(
                            local_output_grad_placements=(Replicate(),)
                        ),
                        "attention.pre_attention.indexer.weights_proj": NoParallel(
                            local_output_grad_placements=(Replicate(),)
                        ),
                        # Upstream NoParallel dropped ``use_local_output``;
                        # output stays a DTensor by default.
                        "attention.pre_attention.indexer.compressor.wkv": NoParallel(),
                        "attention.pre_attention.indexer.compressor.wgate": NoParallel(),
                        "attention.pre_attention.indexer.compressor.norm": NoParallel(),
                    }
                )

        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            # Select the appropriate parallel strategy:
            # Use AwaitRowwiseParallel when activation checkpoint is enabled to handle
            # async redistribution. The custom implementation ensures wait_tensor() is called
            # on _local_tensor to prevent memory leaks caused by incomplete async operations.
            safe_rowwise_parallel = (
                await_rowwise_parallel
                if ac_config.mode in ("full", "selective")
                else rowwise_parallel
            )
            layer_plan.update(
                {
                    "feed_forward": prepare_module_input(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": safe_rowwise_parallel(output_layouts=Shard(1)),
                    "feed_forward.w3": colwise_parallel(),
                }
            )

        # pyrefly: ignore [missing-attribute]
        if transformer_block.layer_id >= model.model_args.n_layers:
            layer_plan.update(
                {
                    "enorm": SequenceParallel(),
                    "hnorm": SequenceParallel(),
                    "e_proj": SequenceParallel(use_local_output=True),
                    "h_proj": SequenceParallel(use_local_output=True),
                }
            )

        parallelize_module(
            # pyrefly: ignore [bad-argument-type]
            module=transformer_block,
            device_mesh=tp_mesh,
            # pyrefly: ignore [bad-argument-type]
            parallelize_plan=layer_plan,
        )

    logger.info("Applied Tensor Parallelism to the model")


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    etp_mesh: DeviceMesh | None,
    ep_etp_mesh: DeviceMesh | None,
    use_deepep: bool = False,
):
    assert (
        tp_mesh is not None or ep_mesh is not None
    ), f"""
        At least one of Tensor Parallel mesh (tp_mesh) or Expert Parallel mesh (ep_mesh) must be provided.
        Current status: tp_mesh={tp_mesh}, ep_mesh={ep_mesh}
        """

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            continue

        if tp_mesh is not None:
            moe_layer_plan = {
                # input / output sharding on the seqlen dim
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1), Replicate()),
                    desired_input_layouts=(Shard(1), Shard(1)),
                    use_local_input=True,
                    output_layouts=(Shard(1),),
                    desired_output_layouts=(Shard(1),),
                ),
                "moe.router.gate": SequenceParallel(
                    sequence_dim=0, use_local_output=True
                ),
            }
            # pyrefly: ignore [missing-attribute]
            if transformer_block.moe.shared_experts is not None:
                # input: sharded on fused batch-seq dimension (dim=0)
                # all-gather for input, reduce-scatter for output
                # pyrefly: ignore [no-matching-overload]
                moe_layer_plan.update(
                    {
                        "moe.shared_experts": PrepareModuleInput(
                            input_layouts=(Shard(0),),
                            desired_input_layouts=(Replicate(),),
                        ),
                        "moe.shared_experts.w1": ColwiseParallel(),
                        "moe.shared_experts.w2": RowwiseParallel(
                            output_layouts=Shard(0)
                        ),
                        "moe.shared_experts.w3": ColwiseParallel(),
                    }
                )
            parallelize_module(
                # pyrefly: ignore [bad-argument-type]
                module=transformer_block,
                device_mesh=tp_mesh,
                # pyrefly: ignore [bad-argument-type]
                parallelize_plan=moe_layer_plan,
            )

        # Currently only TP and TP extend EP are supported
        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            experts_mesh = tp_mesh
            experts_plan = TensorParallel()
        elif tp_mesh is None or etp_mesh is None:
            experts_mesh = ep_mesh
            if use_deepep:
                # pyrefly: ignore [missing-attribute]
                score_before_experts = transformer_block.moe.score_before_experts
                experts_plan = DeepEPExpertParallel(
                    score_before_experts=score_before_experts,
                )
                logger.info("Applying DeepEP to MoE layer")
            else:
                experts_plan = ExpertParallel()
        else:
            raise NotImplementedError("ETP is not supported currently")

        parallelize_module(
            # pyrefly: ignore [missing-attribute]
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )


def _compile_moe_transformer_block(
    transformer_block: nn.Module,
    compile_config,
) -> nn.Module:
    # MoE layers contain FSDP(GroupedExperts) hooks. Compile around those hooks so
    # activation checkpointing does not fall the whole graph back to eager.
    block = (
        transformer_block._checkpoint_wrapped_module
        if isinstance(transformer_block, CheckpointWrapper)
        else transformer_block
    )
    for attr_name, submod in block.named_children():
        if getattr(block, attr_name) != getattr(transformer_block, attr_name):
            raise RuntimeError(
                f"Checkpoint-wrapped block child '{attr_name}' is not exposed on wrapper"
            )
        if attr_name in {"cal_index_loss", "hc_pre"}:
            continue
        if isinstance(submod, moe_module.MoE):
            _compile_children_except(submod, {"experts"}, compile_config)
        elif isinstance(submod, Attention):
            _compile_children_except(submod, {"inner_attention"}, compile_config)
        else:
            setattr(
                block,
                attr_name,
                torch.compile(submod, backend=compile_config.backend, fullgraph=True),
            )
    return transformer_block


def _compile_children_except(
    module: nn.Module,
    skip_names: set[str],
    compile_config,
) -> None:
    for child_name, child_module in module.named_children():
        if child_name in skip_names:
            continue
        setattr(
            module,
            child_name,
            torch.compile(
                child_module,
                backend=compile_config.backend,
                fullgraph=True,
            ),
        )


def apply_compile(model: nn.Module, compile_config, ep_enabled: bool):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    # Required for torch.compile to avoid graph breaking on dynamic shapes in
    # token-choice MoE.
    torch._dynamo.config.capture_scalar_outputs = True
    # Workaround for https://github.com/pytorch/pytorch/issues/166926
    # pyrefly: ignore [missing-attribute]
    torch._C._dynamo.eval_frame._set_lru_cache(False)

    for layer_id, transformer_block in model.layers.named_children():
        if transformer_block.moe_enabled:
            transformer_block = _compile_moe_transformer_block(
                transformer_block, compile_config
            )
        else:
            transformer_block = torch.compile(
                transformer_block,
                backend=compile_config.backend,
                fullgraph=True,
            )
        # pyrefly: ignore [missing-attribute]
        model.layers.register_module(layer_id, transformer_block)

    # NOTE: We don't compile for loop code path due to an issue with unbacked symints:
    # https://github.com/pytorch/pytorch/issues/166460

    logger.info("Compiling each TransformerBlock with torch.compile")


def apply_distributed_indexer_loss_tracking(parallel_dims: ParallelDims):
    """
    Dynamically patch track_dsa_indexer_metrics to support 3D/4D parallelism
    synchronization efficiently using a single global communication step.

    Before synchronization, the indexer loss on each GPU is merely an average
    over its local [B, S] (Batch, Sequence) shape. In a distributed scenario,
    this local loss must be synchronized (averaged) across all parallel domains,
    including Pipeline Parallel (PP), Tensor Parallel (TP), Data Parallel (DP),
    and Context Parallel (CP) groups, to obtain the globally accurate metric.
    """

    # pyrefly: ignore [invalid-decorator]
    @staticmethod
    def distributed_track_dsa_indexer_metrics(total_acc_steps: int):
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return

        # 1. Clone the tensor to avoid modifying the underlying tracker.
        # Shape is always [total_num_layers], so tensor size is consistent across all ranks.
        dsa_indexer_losses = tracker["values"].clone()

        if dist.is_initialized():
            # 2. Perform a SINGLE global All-Reduce (AVG).
            # This averages the tensor across all ranks in the world.
            # For any specific layer, only the ranks in its corresponding PP stage have non-zero values.
            # Therefore, the global sum for a layer is exactly the sum across its non-PP domains.
            dist.all_reduce(dsa_indexer_losses, op=dist.ReduceOp.AVG)

            # 3. Correct the mathematical expectation for Pipeline Parallelism (PP).
            # A global AVG divides the sum by WORLD_SIZE.
            # However, the valid non-zero values only come from (WORLD_SIZE // PP_DEGREE) ranks.
            # By multiplying the result by PP_DEGREE, we recover the exact mathematical
            # average across the non-PP domains (FSDP, TP, CP, etc.).
            pp_degree = parallel_dims.pp if parallel_dims.pp_enabled else 1
            dsa_indexer_losses *= pp_degree

        # 4. Calculate the final aggregated loss.
        # Divide by total gradient accumulation steps to get the true average per step.
        dsa_indexer_num_layers = torch.count_nonzero(dsa_indexer_losses).item()
        loss = dsa_indexer_losses.sum() / dsa_indexer_num_layers / total_acc_steps

        # 5. Clean the tracker and log the metric
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        logger.info(f"indexer loss: {loss.item()}")

    # Apply the monkey patch
    DSAIndexerLossLoggingHelper.track_dsa_indexer_metrics = (
        distributed_track_dsa_indexer_metrics
    )
