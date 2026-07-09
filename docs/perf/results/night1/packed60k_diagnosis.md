# Packed ~60K-token ceiling diagnosis

- generated: 2026-07-07T16:23:37+00:00
- config: `src/track_regression/config/experimental/scaling/finetune_ssm_cls_4L_muon.yaml`
- ckpt: `logs/src/track_regression/logs/comet_offline/1e0f5105c86d4bdd98a0cd3fa780f7dc/ckpts/epoch=049-val_total=0.00125.ckpt`
- env: `{'git_sha': '17a40d1c0fe891ccc7f0f436b84a227cf3bf8548', 'torch': '2.9.1+cu128', 'triton': '3.5.1', 'mamba_ssm': '2.3.0', 'gpu_name': 'NVIDIA H100 NVL', 'hostname': 'sess3', 'cuda_visible_devices': '1'}`
- probe range: 30000..120000 tokens, step 10000, 5 iters/probe

## Probe curve

| tokens (req) | status | classification | detail |
|---:|---|---|---|
| 30000 | ok |  | PROBE_OK tokens=29986 tracks=2295 t_iter_ms=14.00 |
| 40000 | ok |  | PROBE_OK tokens=40010 tracks=3060 t_iter_ms=18.16 |
| 50000 | ok |  | PROBE_OK tokens=50102 tracks=3825 t_iter_ms=22.37 |
| 60000 | ok |  | PROBE_OK tokens=59978 tracks=4590 t_iter_ms=26.47 |
| 70000 | ok |  | PROBE_OK tokens=69988 tracks=5354 t_iter_ms=30.72 |
| 80000 | ok |  | PROBE_OK tokens=79998 tracks=6119 t_iter_ms=34.87 |
| 90000 | ok |  | PROBE_OK tokens=90004 tracks=6884 t_iter_ms=38.97 |
| 100000 | ok |  | PROBE_OK tokens=100008 tracks=7649 t_iter_ms=43.03 |
| 110000 | ok |  | PROBE_OK tokens=109973 tracks=8414 t_iter_ms=47.37 |
| 120000 | ok |  | PROBE_OK tokens=120015 tracks=9179 t_iter_ms=51.55 |

**Last good:** 120000 tokens (requested)  
**First fail:** none within range tokens
