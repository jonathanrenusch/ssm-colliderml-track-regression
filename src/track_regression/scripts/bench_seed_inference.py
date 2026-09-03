#!/usr/bin/env python3
"""Inference-time study: what does the ACTS seed add on top of the model's forward pass?

Times, on one GPU, per batch of packed tracks from a flat store:
  * model forward (TrackParameterRegressor, eval kernels v5pc, fp32 matmul precision as in training)
  * seed on the GPU  (torch twin: padded scatter + select_triplet_torch + estimate_free_torch + perigee)
  * seed residuals on the GPU (P' hit features, seed_residuals_torch)
  * seed on the CPU  (numpy seed_from_csr in the main process, for reference)
and reports us/track and the time per 300 k tracks.

    python scripts/bench_seed_inference.py --run-dir logs/comet_offline/<id> --store /scratch/.../single_muon_uniform/test \
        --batch 10000 --n-tracks 300000 [--gpu 0]
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
os.environ.setdefault("TRK_MATMUL_PRECISION", "highest")
import numpy as np, torch, yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track_regression.flat_data import FlatTrackStore, _gather_block, _pack
from track_regression.seed import seed_from_csr, seed_perigee_torch, seed_residuals_torch
from track_regression.model import TrackRegressionWrapper
from track_regression.mamba_short import apply_variant
from lightning.pytorch.cli import LightningCLI


def build_model(run_dir: Path, ckpt: str, device):
    import tempfile
    full = yaml.safe_load(open(run_dir / "config.yaml"))
    keep = {k: v for k, v in full.items() if k in ("trainer", "model", "data", "seed_everything")}   # drop ckpt_path etc. (subcommand keys)
    keep["trainer"]["logger"] = False; keep["trainer"]["callbacks"] = []
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False); yaml.safe_dump(keep, tmp); tmp.close()
    cfg = tmp.name
    sys.argv = ["bench", "--config", str(cfg), "--trainer.devices", "1"]   # run=False: no subcommand
    from track_regression.data import ColliderMLRegrDataModule
    cli = LightningCLI(model_class=TrackRegressionWrapper, datamodule_class=ColliderMLRegrDataModule,
                       args=sys.argv[1:], run=False, seed_everything_default=42)
    wrapper = cli.model
    sd = torch.load(run_dir / "ckpts" / ckpt, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = wrapper.load_state_dict(sd, strict=False)
    apply_variant(wrapper.model, "v5pc")                    # eval kernels (KernelSwapCallback 'auto' -> v5pc in eval)
    wrapper.model.to(device).eval()
    return wrapper.model, missing, unexpected


def padded_from_packed(H, lens, device):
    """(hits, 12) + lengths -> padded xyz (B,L,3), valid (B,L), vol (B,L) on the GPU."""
    B = lens.numel(); L = int(lens.max())
    starts = torch.cumsum(lens, 0) - lens
    row = torch.repeat_interleave(torch.arange(B, device=device), lens)
    pos = torch.arange(H.shape[0], device=device) - torch.repeat_interleave(starts, lens)
    xyz = torch.zeros(B, L, 3, dtype=torch.float64, device=device); vol = torch.zeros(B, L, dtype=torch.float64, device=device)
    valid = torch.zeros(B, L, dtype=torch.bool, device=device)
    xyz[row, pos] = H[:, :3].double(); vol[row, pos] = H[:, 7].double(); valid[row, pos] = True
    return xyz, valid, vol, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True); ap.add_argument("--ckpt", default="last.ckpt")
    ap.add_argument("--store", required=True); ap.add_argument("--batch", type=int, default=10000)
    ap.add_argument("--n-tracks", type=int, default=300000); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    device = torch.device(f"cuda:{a.gpu}")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model, missing, unexpected = build_model(Path(a.run_dir), a.ckpt, device)
    print(f"model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f} M params; missing {len(missing)}, unexpected {len(unexpected)}")
    store = FlatTrackStore(a.store, load_acts=False)
    n_batches = a.n_tracks // a.batch
    batches = []
    for k in range(n_batches):
        H, T, lens, tg, acts, dm, meta = _gather_block(store, k * a.batch, (k + 1) * a.batch)
        batches.append((H, T, lens, tg))
    ev = lambda: torch.cuda.Event(enable_timing=True)
    t_model = t_seed_gpu = t_res_gpu = t_seed_cpu = t_pack = 0.0; n_done = 0
    with torch.no_grad():
        for k, (H, T, lens, tg) in enumerate(batches):
            # (d) CPU seed, serial numpy (what the collate does today)
            t0 = time.perf_counter(); s_cpu = seed_from_csr(H, lens); t1 = time.perf_counter()
            # collate without timing the seed twice: _pack includes the CPU seed -> subtract
            t2 = time.perf_counter(); inputs, tgt = _pack(H, T, lens, tg); t3 = time.perf_counter()
            inputs = {kk: v.to(device, non_blocking=True) for kk, v in inputs.items()}
            torch.cuda.synchronize()
            Hg = inputs["hit_features"][0]; lens_g = inputs["track_lengths"].long()
            # (a) model forward
            e0, e1 = ev(), ev(); e0.record(); out = model(inputs); e1.record(); torch.cuda.synchronize()
            # (b) GPU seed
            e2, e3 = ev(), ev(); e2.record()
            xyz, valid, vol, row = padded_from_packed(Hg, lens_g, device); s_gpu = seed_perigee_torch(xyz, valid, vol)
            e3.record(); torch.cuda.synchronize()
            # (c) GPU residuals (P' features)
            e4, e5 = ev(), ev(); e4.record(); res = seed_residuals_torch(Hg[:, :3].double(), s_gpu, row); e5.record(); torch.cuda.synchronize()
            if k < a.warmup:
                if k == 0:
                    d = (s_gpu.float().cpu().numpy() - s_cpu); d[:, 2] = np.angle(np.exp(1j * d[:, 2]))
                    print(f"GPU vs CPU seed agreement (batch 0): max |diff| d0 {np.abs(d[:,0]).max():.2e} mm, z0 {np.abs(d[:,1]).max():.2e} mm, phi {np.abs(d[:,2]).max():.2e}, theta {np.abs(d[:,3]).max():.2e}, qop {np.abs(d[:,4]).max():.2e}")
                continue
            t_model += e0.elapsed_time(e1); t_seed_gpu += e2.elapsed_time(e3); t_res_gpu += e4.elapsed_time(e5)
            t_seed_cpu += 1e3 * (t1 - t0); t_pack += 1e3 * (t3 - t2); n_done += len(lens)
    per = lambda ms: ms / n_done * 1e3   # us/track
    per300 = lambda ms: ms / n_done * 300_000 / 1e3   # s per 300 k tracks
    print(f"\n{n_done:,} tracks timed in batches of {a.batch:,} ({len(batches) - a.warmup} batches), GPU {torch.cuda.get_device_name(device)}")
    rows = [("model forward (v5pc, fp32)", t_model), ("seed on GPU (torch)", t_seed_gpu), ("seed residuals on GPU (P')", t_res_gpu),
            ("seed on CPU (numpy, serial)", t_seed_cpu), ("full CPU collate incl. seed (_pack)", t_pack)]
    print(f"{'component':40s} {'us/track':>9s} {'s / 300k tracks':>16s} {'% of model':>11s}")
    for name, ms in rows:
        print(f"{name:40s} {per(ms):9.3f} {per300(ms):16.3f} {100*ms/max(t_model,1e-9):10.1f}%")
    print(f"\n=> inference with the seed on the GPU: model + seed = {per300(t_model + t_seed_gpu):.3f} s / 300k (+{100*t_seed_gpu/t_model:.1f} %)"
          f"; with P' residuals too: +{100*(t_seed_gpu + t_res_gpu)/t_model:.1f} %"
          f"; seed on the CPU instead (serial): +{100*t_seed_cpu/t_model:.0f} %")


if __name__ == "__main__":
    main()
