"""Background nvidia-smi sampler for benchmark runs.

``GpuSampler`` is a context manager that spawns a daemon thread issuing one
short-lived ``nvidia-smi --query-gpu=...`` subprocess per sample (no
long-running stream to orphan) and appends each sample as a CSV row.
``summary()`` reduces the samples to the fields logged in every result row.

The ``throttled`` heuristic flags mean SM clocks < 1700 MHz — the H100 NVL
in this pod boosts to 1785 MHz, so a sub-1700 mean during a timed loop means
the 400 W power cap (or thermals) is biting.

Standalone usage::

    pixi run -e default python scripts/perf/gpu_sampler.py \\
        --gpu-index 0 --out /tmp/gpu.csv --duration 10 --interval 1.0
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import threading
import time
from pathlib import Path

QUERY_FIELDS = (
    "timestamp,power.draw,utilization.gpu,utilization.memory,"
    "memory.used,clocks.sm,temperature.gpu"
)
CSV_HEADER = [
    "timestamp", "power_w", "util_gpu_pct", "util_mem_pct",
    "mem_used_mib", "clocks_sm_mhz", "temp_c",
]
THROTTLE_CLOCK_MHZ = 1700.0


class GpuSampler:
    """Sample one GPU via nvidia-smi on a background thread.

    Parameters
    ----------
    gpu_index : int
        Physical GPU index passed to ``nvidia-smi -i`` (NOT remapped by
        ``CUDA_VISIBLE_DEVICES`` — pass the physical index).
    out_csv : str | Path
        Per-sample CSV sink (header written if new/empty).
    interval_s : float
        Sampling period.
    """

    def __init__(self, gpu_index: int, out_csv: str | Path, interval_s: float = 1.0):
        self.gpu_index = int(gpu_index)
        self.out_csv = Path(out_csv)
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict] = []
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "GpuSampler":
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_csv.exists() or self.out_csv.stat().st_size == 0:
            with open(self.out_csv, "w", newline="") as f:
                csv.writer(f).writerow(CSV_HEADER)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="GpuSampler")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 10.0)
            self._thread = None

    def __enter__(self) -> "GpuSampler":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # -- internals ------------------------------------------------------------

    def _query_once(self) -> dict | None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={QUERY_FIELDS}",
                    "--format=csv,noheader,nounits",
                    "-i", str(self.gpu_index),
                ],
                capture_output=True, text=True, timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        line = out.stdout.strip().splitlines()
        if out.returncode != 0 or not line:
            return None
        parts = [p.strip() for p in line[0].split(",")]
        if len(parts) != 7:
            return None

        def _f(s: str) -> float | None:
            try:
                return float(s)
            except ValueError:
                return None  # "[N/A]" etc.

        return {
            "timestamp": parts[0],
            "power_w": _f(parts[1]),
            "util_gpu_pct": _f(parts[2]),
            "util_mem_pct": _f(parts[3]),
            "mem_used_mib": _f(parts[4]),
            "clocks_sm_mhz": _f(parts[5]),
            "temp_c": _f(parts[6]),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            sample = self._query_once()
            if sample is not None:
                with self._lock:
                    self._samples.append(sample)
                try:
                    with open(self.out_csv, "a", newline="") as f:
                        csv.writer(f).writerow([sample[k2] for k2 in (
                            "timestamp", "power_w", "util_gpu_pct", "util_mem_pct",
                            "mem_used_mib", "clocks_sm_mhz", "temp_c",
                        )])
                except OSError:
                    pass
            # Sleep the remainder of the interval, waking early on stop().
            remaining = self.interval_s - (time.monotonic() - t0)
            if remaining > 0:
                self._stop.wait(remaining)

    # -- reduction ------------------------------------------------------------

    def summary(self) -> dict:
        """Reduce collected samples to the standard result-row fields."""
        with self._lock:
            samples = list(self._samples)

        def col(key: str) -> list[float]:
            return [s[key] for s in samples if s.get(key) is not None]

        def mean(xs: list[float]) -> float | None:
            return sum(xs) / len(xs) if xs else None

        power = col("power_w")
        clocks = col("clocks_sm_mhz")
        clocks_mean = mean(clocks)
        return {
            "power_w_mean": mean(power),
            "power_w_max": max(power) if power else None,
            "sm_util_mean": mean(col("util_gpu_pct")),
            "vram_mib_max": max(col("mem_used_mib"), default=None),
            "clocks_sm_mean": clocks_mean,
            "n_samples": len(samples),
            "throttled": bool(clocks_mean is not None and clocks_mean < THROTTLE_CLOCK_MHZ),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu-index", type=int, default=0, help="physical GPU index for nvidia-smi -i")
    ap.add_argument("--out", type=Path, required=True, help="per-sample CSV path")
    ap.add_argument("--duration", type=float, default=10.0, help="seconds to sample")
    ap.add_argument("--interval", type=float, default=1.0, help="sampling period [s]")
    args = ap.parse_args()

    with GpuSampler(args.gpu_index, args.out, args.interval) as sampler:
        time.sleep(args.duration)
    import json

    print(json.dumps(sampler.summary(), indent=2))


if __name__ == "__main__":
    main()
