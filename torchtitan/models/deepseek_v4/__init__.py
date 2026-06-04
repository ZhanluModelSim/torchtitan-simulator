from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.protocols.model_spec import ModelSpec

from .model import DeepSeekV4Model
from .moe import MoEArgs
from .parallelize import parallelize_deepseek_v4
from .state_dict_adapter import DeepSeekV4StateDictAdapter


def _make_moe_args():
    return MoEArgs(
        num_experts=256, num_shared_experts=1, top_k=6,
        score_func="sqrtsoftplus", route_norm=True,
        score_before_experts=False, use_grouped_mm=True,
        n_hash_layers=3, swiglu_limit=10,
    )


def _make_smoke_moe_args():
    return MoEArgs(
        num_experts=8, num_shared_experts=1, top_k=2,
        num_expert_groups=2, num_limited_groups=2,
        score_func="sqrtsoftplus", route_norm=True,
        score_before_experts=False, use_grouped_mm=False,
        n_hash_layers=0, swiglu_limit=10,
    )


def _smoketest_model():
    return DeepSeekV4Model.Config(
        vocab_size=129280, n_layers=2, n_heads=4,
        max_batch_size=2, max_seq_len=128, dim=128,
        moe_inter_dim=64, head_dim=32, rope_head_dim=16,
        q_lora_rank=64, o_lora_rank=32, o_groups=4,
        window_size=32, compress_ratios=(1, 1),
        moe_args=_make_smoke_moe_args(),
        hc_sinkhorn_iters=4, hc_mult=2, hc_eps=1e-6,
        compress_rope_theta=40000, original_seq_len=128,
        rope_theta=10000, rope_factor=4, beta_fast=32, beta_slow=1,
        enable_indexer_loss=False, index_n_heads=4,
        index_head_dim=16, index_topk=16,
        save_format="hf", save_expert_format=None, hf_save_dir=None,
    )


def _285b_debug_4_layers():
    return DeepSeekV4Model.Config(
        vocab_size=129280, n_layers=4, n_heads=64,
        max_batch_size=4, max_seq_len=4096, dim=4096,
        moe_inter_dim=2048, head_dim=512, rope_head_dim=64,
        q_lora_rank=1024, o_lora_rank=1024, o_groups=8,
        window_size=128, compress_ratios=(1, 1, 4, 128),
        moe_args=_make_moe_args(),
        hc_sinkhorn_iters=4, hc_mult=2, hc_eps=1e-6,
        compress_rope_theta=40000, original_seq_len=128,
        rope_theta=10000, rope_factor=4, beta_fast=32, beta_slow=1,
        enable_indexer_loss=False, index_n_heads=4,
        index_head_dim=16, index_topk=16,
        save_format="hf", save_expert_format=None, hf_save_dir=None,
    )


deepseek_v4_configs = {
    "smoketest": _smoketest_model,
    "debugmodel": _smoketest_model,
    "285b_debug_4_layers": _285b_debug_4_layers,
}


def model_registry(flavor: str, **kwargs) -> ModelSpec:
    config = deepseek_v4_configs[flavor]()
    return ModelSpec(
        name="deepseek_v4", flavor=flavor, model=config,
        parallelize_fn=parallelize_deepseek_v4,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=DeepSeekV4StateDictAdapter,
    )
