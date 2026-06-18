import re

with open('torchtitan/experiments/simulator/trainer_runner.py', 'r') as f:
    content = f.read()

# We want to replace run_trainer_simulation with our new wrapper.
# Find where run_trainer_simulation starts
start_idx = content.find("def run_trainer_simulation(trainer: Any, sim_opts: Any) -> None:")

if start_idx != -1:
    content = content[:start_idx] + """def run_trainer_simulation(trainer: Any, sim_opts: Any) -> None:
    \"\"\"Run one simulated training step seamlessly utilizing the native Trainer.\"\"\"
    from torchtitan.trainer import Trainer
    import torchtitan.observability.structured_logger as sl
    import torchtitan.distributed.utils as dist_utils
    import torch._subclasses.fake_impls
    
    # 1. Patch meta vs meta:0 issues by enforcing clean meta device
    trainer.device = torch.device("meta")
    
    # 2. Patch FakeTensor conversions that crash native train_step
    def _mock_local_scalar_dense(fake_mode, func, *args, **kwargs):
        return 0
    torch._subclasses.fake_impls.op_implementations_dict[torch.ops.aten._local_scalar_dense.default] = _mock_local_scalar_dense
    
    # 3. Patch distributed and optimizer operations that expect real tensors
    orig_clip_grad_norm = dist_utils.clip_grad_norm_
    orig_dist_sum = dist_utils.dist_sum
    orig_dist_max = dist_utils.dist_max
    
    dist_utils.clip_grad_norm_ = lambda *args, **kwargs: torch.tensor(0.0, device="meta")
    dist_utils.dist_sum = lambda t, *args, **kwargs: t
    dist_utils.dist_max = lambda t, *args, **kwargs: t
    
    orig_get_mesh = trainer.parallel_dims.get_optional_mesh
    trainer.parallel_dims.get_optional_mesh = lambda *args, **kwargs: None

    orig_optim_step = trainer.optimizers.step
    orig_lr_step = trainer.lr_schedulers.step
    trainer.optimizers.step = lambda *args, **kwargs: None
    trainer.lr_schedulers.step = lambda *args, **kwargs: None

    # Patch log_trace_scalar to avoid int() crash on meta tensors
    orig_log = sl.log_trace_scalar
    def safe_log(d):
        safe_dict = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                safe_dict[k] = 0
            else:
                try:
                    safe_dict[k] = int(v)
                except Exception:
                    safe_dict[k] = 0
        orig_log(safe_dict)
    sl.log_trace_scalar = safe_log

    recorder = TraceRecorder(rank=int(os.environ.get("RANK", "0")))
    
    data_iterator = trainer.batch_generator(trainer.dataloader)
    
    # Pre-fetch batches outside of FakeTensorMode to avoid dataloader internal crashes
    batches = []
    for _ in range(trainer.gradient_accumulation_steps):
        batches.append(next(data_iterator))
        
    def mock_data_iterator():
        for batch in batches:
            yield batch
    
    use_fake = (getattr(sim_opts, "comm_backend", "") or "") != "gloo"

    try:
        with unified_trace(recorder, use_fake_mode=use_fake, capture_comm=not use_fake, capture_fsdp=not use_fake):
            Trainer.train_step(trainer, mock_data_iterator())
    finally:
        # Restore patched methods
        dist_utils.clip_grad_norm_ = orig_clip_grad_norm
        dist_utils.dist_sum = orig_dist_sum
        dist_utils.dist_max = orig_dist_max
        trainer.parallel_dims.get_optional_mesh = orig_get_mesh
        trainer.optimizers.step = orig_optim_step
        trainer.lr_schedulers.step = orig_lr_step
        sl.log_trace_scalar = orig_log

    result = recorder.build_result()

    if use_fake:
        _inject_synthetic_comm_events(result, trainer, sim_opts)
    if getattr(sim_opts, "semantic_schedule", False):
        _inject_semantic_schedule(result, trainer.config)

    if getattr(sim_opts, "cost_model", False):
        cm = _import_cost_model(
            getattr(sim_opts, "cost_model_class", "") or "torchtitan.experiments.simulator.cost_model.MockCostModel",
            _get_cost_model_kwargs(sim_opts),
        )
        apply_cost_model(result, cm)

    postprocess_extension_result(trainer, result)
    _export_result(result, sim_opts.output_dir, sim_opts.output_formats)
"""

with open('torchtitan/experiments/simulator/trainer_runner.py', 'w') as f:
    f.write(content)
