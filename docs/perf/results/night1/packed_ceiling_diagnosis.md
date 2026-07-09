# Packed ~60K-token ceiling diagnosis

- generated: 2026-07-07T18:33:00+00:00
- config: `src/track_regression/config/experimental/scaling/finetune_ssm_cls_4L_muon.yaml`
- ckpt: `logs/src/track_regression/logs/comet_offline/1e0f5105c86d4bdd98a0cd3fa780f7dc/ckpts/epoch=049-val_total=0.00125.ckpt`
- env: `{'git_sha': '17a40d1c0fe891ccc7f0f436b84a227cf3bf8548', 'torch': '2.9.1+cu128', 'triton': '3.5.1', 'mamba_ssm': '2.3.0', 'gpu_name': 'NVIDIA H100 NVL', 'hostname': 'sess3', 'cuda_visible_devices': '1'}`
- probe range: 860000..940000 tokens, step 10000, 5 iters/probe

## Probe curve

| tokens (req) | status | classification | detail |
|---:|---|---|---|
| 860000 | ok |  | PROBE_OK tokens=859980 tracks=65783 t_iter_ms=362.53 |
| 870000 | ok |  | PROBE_OK tokens=870004 tracks=66548 t_iter_ms=367.07 |
| 880000 | ok |  | PROBE_OK tokens=880096 tracks=67313 t_iter_ms=371.25 |
| 890000 | ok |  | PROBE_OK tokens=889972 tracks=68078 t_iter_ms=375.28 |
| 900000 | ok |  | PROBE_OK tokens=899995 tracks=68843 t_iter_ms=379.51 |
| 910000 | FAIL | launch-error | PROBE_STATS tracks=69608 tokens=910004 len_mean=13.073 |
| 905000 | ok |  | PROBE_OK tokens=905057 tracks=69225 t_iter_ms=381.61 |
| 907500 | ok |  | PROBE_OK tokens=907571 tracks=69417 t_iter_ms=382.87 |
| 908750 | ok |  | PROBE_OK tokens=908792 tracks=69512 t_iter_ms=383.36 |
| 909375 | ok |  | PROBE_OK tokens=909386 tracks=69560 t_iter_ms=382.21 |
| 909687 | FAIL | launch-error | PROBE_STATS tracks=69584 tokens=909699 len_mean=13.073 |

**Last good:** 909375 tokens (requested)  
**First fail:** 909687 tokens

## First failure (909687 tokens, classified: launch-error, rc=1)

PROBE_STATS tracks=69584 tokens=909699 len_mean=13.073

### Traceback (verbatim)

```
Traceback (most recent call last):
  File "/shared/tracking/ssm-colliderml-track-regression/scripts/perf/diagnose_packed_60k.py", line 286, in main
    run_probe(args)
  File "/shared/tracking/ssm-colliderml-track-regression/scripts/perf/diagnose_packed_60k.py", line 93, in run_probe
    model(gpu_inputs)  # warmup / triton compile — outside the timing
    ^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/src/track_regression/model.py", line 572, in forward
    _, pooled = self.encoder(
                ^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/src/track_regression/mamba_cls.py", line 381, in forward
    return self._forward_packed(x, seq_idx, cu_seqlens)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/src/track_regression/mamba_cls.py", line 607, in _forward_packed
    x_aug = layer(x_aug, seq_idx=aug_seq_idx, flip_indices=flip_indices)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/src/track_regression/mamba_state.py", line 338, in forward
    x_fwd = self.forward_mamba(x_norm, seq_idx=seq_idx)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/mamba_ssm/modules/mamba2.py", line 185, in forward
    out = mamba_split_conv1d_scan_combined(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/mamba_ssm/ops/triton/ssd_combined.py", line 997, in mamba_split_conv1d_scan_combined
    return MambaSplitConv1dScanCombinedFn.apply(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states, seq_idx, dt_limit, return_final_states, activation, rmsnorm_weight, rmsnorm_eps, outproj_weight, outproj_bias, headdim, ngroups, norm_before_gate)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/torch/amp/autocast_mode.py", line 527, in decorate_fwd
    return fwd(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/mamba_ssm/ops/triton/ssd_combined.py", line 857, in forward
    out_x, _, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=None, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=dt_limit)
                                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/mamba_ssm/ops/triton/ssd_combined.py", line 374, in _mamba_chunk_scan_combined_fwd
    dA_cumsum, dt = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/mamba_ssm/ops/triton/ssd_chunk_state.py", line 728, in _chunk_cumsum_fwd
    _chunk_cumsum_fwd_kernel[grid_chunk_cs](
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/triton/runtime/jit.py", line 419, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 238, in run
    benchmark()
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 227, in benchmark
    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 162, in _bench
    return self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/triton/testing.py", line 149, in do_bench
    fn()
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 148, in kernel_call
    self.fn.run(
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/triton/runtime/jit.py", line 757, in run
    kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
  File "/shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/lib/python3.12/site-packages/triton/backends/nvidia/driver.py", line 712, in __call__
    self.launch(gridX, gridY, gridZ, stream, function, self.launch_cooperative_grid, self.launch_pdl,
RuntimeError: Triton Error [CUDA]: invalid argument
```
