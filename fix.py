import re

with open('torchtitan/experiments/simulator/trainer.py', 'r') as f:
    content = f.read()

# Add back _cpu_noop_parallelize override for simulator so we skip the upstream apply_fsdp which crashes.
# It was removed earlier.
content = content.replace('super().__init__(config)\n\n        # Apply FSDP1', 
'''        if comm_backend == "gloo":
            model_name = getattr(config.model_spec, "name", "")
            if "deepseek" in model_name.lower():
                config.model_spec.parallelize_fn = _cpu_gloo_parallelize_dsv4
            else:
                config.model_spec.parallelize_fn = _cpu_gloo_parallelize_llama
        else:
            config.model_spec.parallelize_fn = _cpu_noop_parallelize

        # Use PP-semantic pipeline when PP > 1 and not gloo mode
        self._pp_model_parts: list[Any] = []
        if pp > 1 and comm_backend != "gloo":
            config.model_spec.pipelining_fn = partial(
                _cpu_semantic_pipeline,
                config=config,
                model_parts_holder=self._pp_model_parts,
            )
        else:
            config.model_spec.pipelining_fn = _cpu_noop_pipeline

        super().__init__(config)
        
        if self._pp_model_parts:
            self.model_parts = self._pp_model_parts

        # Apply FSDP1''')

with open('torchtitan/experiments/simulator/trainer.py', 'w') as f:
    f.write(content)
