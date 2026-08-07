# Overnight-run stability checklist

Environment (verified 2026-07-06): Kubernetes pod `sess3` on a bare-metal CERN
cluster node, accessed from a Mac via VS Code Remote Containers. Pod has been
up 34 days (long-lived — eviction is possible but not the norm). Filesystems:

| Path | Type | Kerberos-gated | Survives pod restart | Use for |
|------|------|----------------|----------------------|---------|
| `/afs/cern.ch/user/<u>/<user>` (home, **`~/.claude` lives here**) | AFS | **YES — token expires daily** | yes | nothing overnight-critical except Claude's own state |
| `/shared` (repo, NFS PVC) | nfs4 | no | yes | **all results, logs, dashboards, queue state** |
| `/scratch` (node NVMe) | xfs | no | node-local (host mount) | dataset reads, big temporaries |
| `/tmp` | overlay | no | **NO** | Triton cache only (rebuilds itself) |

## Before going to bed (every night, in order)

1. **On the Mac:** run `caffeinate -dimsu` in a spare local terminal so the
   laptop never sleeps and the VS Code remote connection stays up.
   (Belt-and-braces: even if it drops, steps 3–4 make the remote side immune.)
2. **Fresh Kerberos + auto-renew (on the pod):**
   ```bash
   kinit <user>@CERN.CH          # fresh 24h ticket (renewable ~5 days)
   aklog                            # fresh AFS token
   krenew -b -t -K 60               # daemon: renew ticket every 60 min AND
                                    # re-run aklog (-t) so AFS never expires
   klist                            # verify: expiry must be > tomorrow morning
   ```
   Why this matters: `~/.claude` (session state, plans, memory) is on AFS. If
   the AFS token lapses overnight, the driving session itself loses its state
   dir. `krenew` keeps both alive; tickets are renewable for ~5 days, so one
   `kinit` at the start of the campaign + nightly checks suffice.
3. **tmux for anything that must survive disconnects:**
   ```bash
   tmux new -s overnight            # or: tmux attach -t overnight
   ```
   Run the overnight driver (Claude CLI session and/or the experiment queue
   runner) INSIDE tmux — not in a VS Code integrated terminal tab. A VS Code
   window reload, Mac sleep, or Wi-Fi blip then costs nothing; reattach in the
   morning with `tmux attach -t overnight`.
4. **Long GPU jobs detached and resumable:** launch benchmarks via the queue
   runner with `nohup … >> /shared/...log 2>&1 &` (or from inside tmux). The
   queue state and all outputs go to `/shared/...` (NFS, no Kerberos), never
   to AFS, so jobs can't block on token expiry mid-run.
5. **Quick GPU sanity:** `nvidia-smi` — both H100s visible, persistence mode
   Enabled (verified), power limit 400 W, no zombie processes hogging VRAM.
6. **Disk headroom:** `df -h /shared /scratch` — /shared was 64 % full
   (184 G free) on 2026-07-06; ncu/nsys reports and HDF5 dumps add up. Keep
   ≥20 G free; park bulky profiles under `/scratch/` if tight.
7. **Data path override in place** (config points at the nonexistent
   `p200_core_kf_hits_finetune`; real dir is `p200_core_kf_matched_finetune`).

## Failure modes & mitigations

- **Mac sleeps / VS Code disconnects** → caffeinate + tmux; remote processes
  unaffected; reattach in the morning.
- **Kerberos/AFS token expiry (default: daily ~15:44)** → `krenew -b -t`;
  additionally nothing on the critical benchmark path reads/writes AFS.
  Fallback if krenew dies: `k5start -f /tmp/krb5cc_<uid> -K 60 -t -b`.
- **Pod eviction/restart (rare but possible on k8s)** → everything resumable:
  queue state + results on `/shared`; `/tmp` Triton cache and any `/scratch`
  temporaries are expendable. After a restart: re-`kinit`, reattach, rerun the
  queue runner — it must skip completed entries (design requirement).
- **One benchmark hangs** → per-job timeout in the queue runner + heartbeat
  line in the log; a hung job must not stall the whole night (skip & flag).
- **OOM/crash cascade** → each experiment runs as its own subprocess; a crash
  is recorded as a result (status=failed) and the runner moves on.

## Morning routine (user or assistant)

```bash
tmux attach -t overnight   # or: tmux ls
klist                      # ticket still fresh?
tail -50 /shared/tracking/ssm-colliderml-track-regression/docs/perf/results/night_run.log
nvidia-smi                 # anything still running / wedged?
```
Then read the nightly section appended to `docs/perf/OPTIMIZATION_LOG.md`.
