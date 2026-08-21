"""Overnight queue runner: two GPU lanes, subprocess-isolated, resumable.

Reads a YAML queue file (see ``scripts/perf/queues/night1.yaml``)::

    night: 1
    defaults: {timeout_s: 3600}
    jobs:
      - id: v0_staged
        kind: bench            # bench | sweep | profile_nsys | profile_ncu | physics | custom
        variant: v0
        gpu: any               # any | 0 | 1 | exclusive
        heavy_cpu: false
        depends_on: []
        timeout_s: 3600
        env: default           # default | mamba232 (pixi env)
        args: {config: ..., ckpt: ..., mode: staged, batch-size: 22000}
      - id: subset_build
        kind: custom
        cmd: "pixi run -e default python scripts/perf/physics_drift.py ..."

Behaviour
---------
- Two lane threads (gpu0 / gpu1); each job is a ``subprocess.Popen`` with
  ``start_new_session=True``, ``CUDA_VISIBLE_DEVICES=<lane>``,
  ``TRITON_CACHE_DIR=/tmp/triton_cache``, stdout+stderr →
  ``<results-dir>/jobs/<id>.log``.
- States: pending / running / done / failed / timeout / interrupted
  (interrupted jobs are requeued once; at most 2 attempts total).
- ``queue_state.json`` is atomically snapshotted on every transition and
  ``events.jsonl`` appended; a human one-liner goes to
  ``<results-dir>/night_run.log`` AND ``<results-dir>/../night_run.log``.
- ``gpu: exclusive`` jobs acquire both lanes; ``heavy_cpu: true`` jobs are
  mutually exclusive with each other. 60 s stagger between lane starts.
- Poll every 15 s; a job is killed (whole process group) on timeout or when
  its log file goes stale for > 600 s.
- Resumable: re-run the same command — done jobs are skipped, jobs found
  "running" in the state file are requeued as interrupted. The queue file is
  re-read every scheduling pass, so new jobs may be appended mid-night.

Usage::

    pixi run -e default python scripts/perf/runner.py \\
        --queue scripts/perf/queues/night1.yaml \\
        --results-dir docs/perf/results/night1
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

REPO_ROOT = common.REPO_ROOT
POLL_S = 15.0
STAGGER_S = 60.0
STALE_LOG_S = 600.0
MAX_ATTEMPTS = 2
TERMINAL = {"done", "failed", "timeout"}


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class Runner:
    def __init__(self, queue_path: Path, results_dir: Path,
                 poll_s: float = POLL_S, stagger_s: float = STAGGER_S):
        self.poll_s = poll_s
        self.stagger_s = stagger_s
        self.queue_path = queue_path
        self.results_dir = results_dir
        self.jobs_dir = results_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = results_dir / "queue_state.json"
        self.events_path = results_dir / "events.jsonl"

        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.queue_mtime = 0.0
        self.queue_defaults: dict = {}
        self.job_specs: dict[str, dict] = {}   # id -> spec (queue order preserved)
        self.state: dict[str, dict] = {}       # id -> {status, attempts, ...}
        self.lane_current: dict[int, str | None] = {0: None, 1: None}
        self.exclusive_holder: str | None = None
        self.heavy_cpu_holder: str | None = None
        self.last_start_ts = 0.0
        self.procs: dict[str, subprocess.Popen] = {}

        self._load_state()
        self._reload_queue(force=True)
        # Anything found "running" was orphaned by a previous runner death.
        for jid, st in self.state.items():
            if st.get("status") == "running":
                self._transition(jid, "interrupted", reason="found running at startup")
        self._requeue_interrupted()
        self._snapshot()

    # -- persistence ---------------------------------------------------------

    def _load_state(self) -> None:
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    self.state = json.load(f).get("jobs", {})
                print(f"[runner] resumed state for {len(self.state)} jobs", flush=True)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[runner] WARNING: could not load state: {e}", flush=True)
                self.state = {}

    def _snapshot(self) -> None:
        with self.lock:
            common.atomic_write_json(self.state_path, {
                "updated": _now(),
                "queue": str(self.queue_path),
                "jobs": self.state,
            })

    def _log_line(self, msg: str) -> None:
        line = f"{_now()} {msg}"
        print(f"[runner] {msg}", flush=True)
        for p in (self.results_dir / "night_run.log",
                  self.results_dir.parent / "night_run.log"):
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "a") as f:
                    f.write(line + "\n")
            except OSError:
                pass

    def _transition(self, jid: str, status: str, **extra) -> None:
        with self.lock:
            st = self.state.setdefault(jid, {"status": "pending", "attempts": 0})
            prev = st.get("status")
            st["status"] = status
            st["updated"] = _now()
            st.update(extra)
            event = {"ts": _now(), "job": jid, "from": prev, "to": status, **extra}
            common.append_jsonl(self.events_path, event)
            self._snapshot()
        detail = " ".join(
            f"{k}={v}" for k, v in extra.items() if k != "cmd" and v not in (None, "")
        )
        self._log_line(f"[{jid}] {prev} -> {status}" + (f" ({detail})" if detail else ""))

    def _requeue_interrupted(self) -> None:
        for jid, st in self.state.items():
            if st.get("status") == "interrupted":
                if st.get("attempts", 0) < MAX_ATTEMPTS:
                    self._transition(jid, "pending", reason="requeued after interrupt")
                else:
                    self._transition(jid, "failed", reason="max attempts after interrupt")

    # -- queue file ------------------------------------------------------------

    def _reload_queue(self, force: bool = False) -> None:
        try:
            mtime = self.queue_path.stat().st_mtime
        except OSError:
            return
        if not force and mtime == self.queue_mtime:
            return
        try:
            with open(self.queue_path) as f:
                q = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            print(f"[runner] WARNING: bad queue file: {e}", flush=True)
            return
        self.queue_mtime = mtime
        self.queue_defaults = q.get("defaults", {}) or {}
        new_specs: dict[str, dict] = {}
        for spec in q.get("jobs", []) or []:
            jid = spec.get("id")
            if not jid:
                continue
            new_specs[jid] = spec
            if jid not in self.state:
                self.state[jid] = {"status": "pending", "attempts": 0}
                self._log_line(f"[{jid}] registered (pending)")
        self.job_specs = new_specs

    # -- command construction ---------------------------------------------------

    def _build_cmd(self, spec: dict) -> str:
        kind = spec.get("kind", "bench")
        env_name = spec.get("env", "default")
        if kind == "custom" or "cmd" in spec:
            return spec["cmd"]
        if kind in ("profile_nsys", "profile_ncu"):
            raise ValueError(f"kind={kind} requires an explicit 'cmd'")
        script = {
            "bench": "scripts/perf/bench_variant.py",
            "sweep": "scripts/perf/bench_variant.py",
            "physics": "scripts/perf/physics_drift.py",
        }.get(kind)
        if script is None:
            raise ValueError(f"unknown job kind: {kind}")

        args = dict(spec.get("args", {}) or {})
        if kind == "sweep":
            args.setdefault("mode", "sweep")
        if spec.get("variant") and "variant" not in args:
            args["variant"] = spec["variant"]
        if script.endswith("bench_variant.py"):
            args.setdefault("job-id", spec["id"])
            args.setdefault("out-jsonl", str(self.results_dir / "results.jsonl"))
            args.setdefault("gpu-samples-csv", str(self.jobs_dir / f"{spec['id']}_gpu.csv"))

        parts = ["pixi", "run", "-e", env_name, "python", script]
        for key, val in args.items():
            flag = "--" + str(key).replace("_", "-")
            if isinstance(val, bool):
                if val:
                    parts.append(flag)
            elif isinstance(val, list):
                for v in val:
                    parts += [flag, str(v)]
            else:
                parts += [flag, str(val)]
        return " ".join(shlex.quote(p) for p in parts)

    # -- scheduling ---------------------------------------------------------------

    def _dep_state(self, spec: dict) -> str:
        """'ready' if all deps done, 'dead' if any dep terminally failed, else 'wait'."""
        for dep in spec.get("depends_on", []) or []:
            st = self.state.get(dep, {}).get("status")
            if st == "done":
                continue
            if st in ("failed", "timeout"):
                return "dead"
            return "wait"
        return "ready"

    def _claim(self, jid: str, spec: dict, lane: int, exclusive: bool) -> dict:
        """Mark lane(s)/locks as held by jid. Caller holds self.lock."""
        if exclusive:
            self.exclusive_holder = jid
            self.lane_current[1 - lane] = jid
        if spec.get("heavy_cpu"):
            self.heavy_cpu_holder = jid
        self.lane_current[lane] = jid
        self.last_start_ts = time.time()
        st = self.state.setdefault(jid, {"status": "pending", "attempts": 0})
        st["attempts"] = st.get("attempts", 0) + 1
        return spec

    def _acquire(self, lane: int) -> dict | None:
        """Pick and claim the next runnable job for a lane (or None)."""
        with self.lock:
            self._reload_queue()
            if self.lane_current[lane] is not None:
                return None
            other = 1 - lane

            # An exclusive job reserved the queue: only start it when both
            # lanes are idle; block all other starts meanwhile.
            if self.exclusive_holder is not None:
                jid = self.exclusive_holder
                spec = self.job_specs.get(jid)
                if spec is None or self.state.get(jid, {}).get("status") != "pending":
                    self.exclusive_holder = None  # stale reservation
                    return None
                if self.lane_current[other] is None:
                    return self._claim(jid, spec, lane, exclusive=True)
                return None

            if self.last_start_ts > 0 and time.time() - self.last_start_ts < self.stagger_s:
                return None

            for jid, spec in self.job_specs.items():
                st = self.state.get(jid, {})
                if st.get("status") != "pending":
                    continue
                if st.get("attempts", 0) >= MAX_ATTEMPTS:
                    self._transition(jid, "failed", reason="max attempts")
                    continue
                dep = self._dep_state(spec)
                if dep == "dead":
                    self._transition(jid, "failed", reason="dependency failed")
                    continue
                if dep == "wait":
                    continue
                gpu = str(spec.get("gpu", "any"))
                if gpu not in ("any", "exclusive") and int(gpu) != lane:
                    continue
                if gpu == "exclusive":
                    if self.lane_current[other] is not None:
                        # Reserve: block new starts until the other lane drains.
                        self.exclusive_holder = jid
                        return None
                    return self._claim(jid, spec, lane, exclusive=True)
                if spec.get("heavy_cpu") and self.heavy_cpu_holder is not None:
                    continue
                return self._claim(jid, spec, lane, exclusive=False)
            return None

    def _release(self, jid: str, lane: int) -> None:
        with self.lock:
            for ln in (0, 1):
                if self.lane_current[ln] == jid:
                    self.lane_current[ln] = None
            if self.exclusive_holder == jid:
                self.exclusive_holder = None
            if self.heavy_cpu_holder == jid:
                self.heavy_cpu_holder = None
            self.procs.pop(jid, None)

    # -- job execution -----------------------------------------------------------

    def _kill_group(self, proc: subprocess.Popen) -> None:
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return
        for sig, grace in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            deadline = time.time() + grace
            while time.time() < deadline:
                if proc.poll() is not None:
                    return
                time.sleep(0.5)

    def _run_job(self, spec: dict, lane: int) -> None:
        jid = spec["id"]
        timeout_s = float(spec.get("timeout_s", self.queue_defaults.get("timeout_s", 7200)))
        exclusive = str(spec.get("gpu", "any")) == "exclusive"
        try:
            cmd = self._build_cmd(spec)
        except (ValueError, KeyError) as e:
            self._transition(jid, "failed", reason=f"bad spec: {e}")
            self._release(jid, lane)
            return

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1" if exclusive else str(lane)
        env["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
        log_path = self.jobs_dir / f"{jid}.log"

        with open(log_path, "a") as logf:
            logf.write(f"\n===== {_now()} lane={lane} attempt={self.state[jid]['attempts']} =====\n$ {cmd}\n")
            logf.flush()
            try:
                proc = subprocess.Popen(
                    cmd, shell=True, cwd=REPO_ROOT, env=env,
                    stdout=logf, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as e:
                self._transition(jid, "failed", reason=f"spawn failed: {e}")
                self._release(jid, lane)
                return

            with self.lock:
                self.procs[jid] = proc
            self._transition(jid, "running", lane=lane, pid=proc.pid, cmd=cmd)

            started = time.time()
            status, reason = None, ""
            while True:
                rc = proc.poll()
                if rc is not None:
                    status = "done" if rc == 0 else "failed"
                    reason = f"rc={rc}"
                    break
                if self.stop_event.is_set():
                    self._kill_group(proc)
                    status, reason = "interrupted", "runner stopped"
                    break
                elapsed = time.time() - started
                if elapsed > timeout_s:
                    self._kill_group(proc)
                    status, reason = "timeout", f"timeout after {elapsed:.0f}s"
                    break
                try:
                    stale = time.time() - log_path.stat().st_mtime
                except OSError:
                    stale = 0.0
                # Profiler jobs (ncu kernel-replay in particular) legitimately
                # produce no output for many minutes — only the hard timeout
                # applies to them.
                is_profiler = str(spec.get("kind", "")).startswith("profile_")
                if stale > STALE_LOG_S and not is_profiler:
                    self._kill_group(proc)
                    status, reason = "timeout", f"log stale for {stale:.0f}s"
                    break
                self.stop_event.wait(self.poll_s)

        rc = proc.poll()
        self._transition(jid, status, reason=reason, rc=rc,
                         runtime_s=round(time.time() - started, 1))
        self._release(jid, lane)

    # -- lanes -------------------------------------------------------------------

    def _lane_loop(self, lane: int) -> None:
        while not self.stop_event.is_set():
            spec = self._acquire(lane)
            if spec is None:
                if self._all_terminal():
                    return
                self.stop_event.wait(5.0)
                continue
            self._run_job(spec, lane)

    def _all_terminal(self) -> bool:
        with self.lock:
            if any(j is not None for j in self.lane_current.values()):
                return False
            return all(
                self.state.get(jid, {}).get("status") in TERMINAL
                for jid in self.job_specs
            )

    # -- entry -------------------------------------------------------------------

    def run(self) -> int:
        self._log_line(f"runner start queue={self.queue_path} results={self.results_dir}")
        threads = [
            threading.Thread(target=self._lane_loop, args=(lane,), name=f"lane{lane}", daemon=True)
            for lane in (0, 1)
        ]
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=1.0)
        except KeyboardInterrupt:
            self._log_line("Ctrl-C: stopping — terminating children, saving state")
            self.stop_event.set()
            for t in threads:
                t.join(timeout=30.0)
        self._snapshot()
        counts: dict[str, int] = {}
        for jid in self.job_specs:
            s = self.state.get(jid, {}).get("status", "unknown")
            counts[s] = counts.get(s, 0) + 1
        self._log_line(f"runner exit: {counts}")
        return 0 if counts.get("failed", 0) == 0 and counts.get("timeout", 0) == 0 else 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--queue", required=True, type=Path, help="queue YAML file")
    ap.add_argument("--results-dir", required=True, type=Path,
                    help="e.g. docs/perf/results/night1")
    ap.add_argument("--poll-s", type=float, default=POLL_S,
                    help="subprocess poll period")
    ap.add_argument("--stagger-s", type=float, default=STAGGER_S,
                    help="minimum delay between lane starts")
    args = ap.parse_args()

    # Route SIGTERM through KeyboardInterrupt for a clean shutdown path.
    def _sigterm(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm)

    runner = Runner(args.queue.resolve(), args.results_dir.resolve(),
                    poll_s=args.poll_s, stagger_s=args.stagger_s)
    sys.exit(runner.run())


if __name__ == "__main__":
    main()
