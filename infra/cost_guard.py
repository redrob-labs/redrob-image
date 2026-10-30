#!/usr/bin/env python3
"""Cost guard for a training instance. Run ON the instance next to the trainer (systemd unit or nohup).

Stops the run when the wall-clock budget or the dollar budget is spent:
  1. SIGTERM to the trainer pid (scripts/train_lora.py checkpoints and exits on SIGTERM),
  2. waits --grace-min for it to exit (checkpoint upload included),
  3. powers the instance off (`sudo shutdown -h now`; with InstanceInitiatedShutdownBehavior=terminate,
     or at the end of a Capacity Block, the instance is gone).
Writes a line to --log every check so the decision is auditable.

    python infra/cost_guard.py --pid <trainer pid> --usd-per-hour 31.0 --budget-usd 4000 --max-hours 120 \
        --log ~/runs/<name>/cost_guard.log            # add --dry-run to never power off
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time


def alive(pid: int) -> bool:
    return os.path.isdir(f"/proc/{pid}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--usd-per-hour", type=float, required=True)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--max-hours", type=float, required=True)
    ap.add_argument("--started", type=float, default=0.0, help="unix time the instance started (default: now)")
    ap.add_argument("--until", default="", help="Capacity Block end, UTC ISO (e.g. 2026-11-03T11:30:00Z). EC2 starts "
                    "terminating 30 min before it, so the guard stops the trainer at end - 30 min - grace - 10 min")
    ap.add_argument("--grace-min", type=float, default=20)
    ap.add_argument("--every-sec", type=int, default=60)
    ap.add_argument("--log", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    t0 = a.started or time.time()
    limit_h = min(a.max_hours, a.budget_usd / a.usd_per_hour)
    if a.until:
        import calendar

        end = calendar.timegm(time.strptime(a.until.replace("Z", ""), "%Y-%m-%dT%H:%M:%S"))
        limit_h = min(limit_h, (end - (30 + a.grace_min + 10) * 60 - t0) / 3600)
    with open(os.path.expanduser(a.log), "a", encoding="utf-8") as log:
        def note(msg):
            log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")
            log.flush()

        note(f"guard start: limit {limit_h:.2f} h (max {a.max_hours} h, ${a.budget_usd} at ${a.usd_per_hour}/h)")
        while True:
            h = (time.time() - t0) / 3600
            if not alive(a.pid):
                note(f"trainer exited after {h:.2f} h, ${h * a.usd_per_hour:.0f}; powering off")
                break
            if h >= limit_h:
                note(f"budget reached at {h:.2f} h, ${h * a.usd_per_hour:.0f}; SIGTERM {a.pid}")
                os.kill(a.pid, signal.SIGTERM)
                end = time.time() + a.grace_min * 60
                while alive(a.pid) and time.time() < end:
                    time.sleep(10)
                note("trainer gone" if not alive(a.pid) else "grace over, trainer still running; powering off anyway")
                break
            time.sleep(a.every_sec)
        if a.dry_run:
            note("dry-run: not powering off")
            return 0
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
