#!/usr/bin/env python3
"""Stop only this recipe's vLLM unit before host memory is exhausted."""
import json
import subprocess
import sys
import time
from pathlib import Path

UNIT = sys.argv[1] if len(sys.argv) == 2 else None
if not UNIT or not UNIT.endswith('.service') or '/' in UNIT:
    raise SystemExit('usage: guard.py <owned-unit.service>')
LOG = Path.home() / '.local/state/model-serving-recipes' / (UNIT + '.jsonl')
LOG.parent.mkdir(parents=True, exist_ok=True)
MEM_FLOOR = 4 * 1024 * 1024  # kB; preserve at least 2 GiB outside the guard.
SWAP_FLOOR = 2 * 1024 * 1024


def memory():
    result = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(('MemAvailable:', 'SwapFree:')):
            k, v = line.split(':', 1)
            result[k] = int(v.split()[0])
    return result


def state(unit):
    p = subprocess.run(['systemctl', '--user', 'is-active', unit],
                       capture_output=True, text=True, timeout=5)
    return p.stdout.strip()


def record(event, **fields):
    item = {'epoch': time.time(), 'event': event, **fields}
    with LOG.open('a') as f:
        f.write(json.dumps(item) + '\n')
    print(json.dumps(item), flush=True)


def stop_owned(reason, current):
    record('guard-trigger', reason=reason, current=current, unit=UNIT)
    subprocess.run(['systemctl', '--user', 'stop', '--no-block', UNIT],
                   capture_output=True, timeout=5)
    for _ in range(20):
        if state(UNIT) not in ('active', 'activating', 'deactivating'):
            break
        time.sleep(0.5)
    if state(UNIT) in ('active', 'activating', 'deactivating'):
        subprocess.run(['systemctl', '--user', 'kill', '--signal=SIGKILL', UNIT],
                       capture_output=True, timeout=5)
    record('guard-stop', reason=reason, current=memory(), unit_state=state(UNIT))
    raise SystemExit(2)


initial = memory()
if initial['MemAvailable'] < MEM_FLOOR or initial['SwapFree'] < SWAP_FLOOR:
    raise SystemExit('insufficient host memory/swap for guarded launch')
manager_was_active = state('rexl3-manager.service') == 'active'
record('started', baseline=initial, manager_was_active=manager_was_active)
started = time.monotonic()
seen = False
min_available = initial['MemAvailable']
min_swap_free = initial['SwapFree']
tick = 0
while True:
    now = memory()
    min_available = min(min_available, now['MemAvailable'])
    min_swap_free = min(min_swap_free, now['SwapFree'])
    manager = state('rexl3-manager.service')
    if now['MemAvailable'] < MEM_FLOOR:
        stop_owned('available-under-4GiB', now)
    if now['SwapFree'] < SWAP_FLOOR:
        stop_owned('swap-free-under-2GiB', now)
    if manager_was_active and manager != 'active':
        stop_owned('manager-exited', now)
    current = state(UNIT)
    if current == 'active':
        seen = True
    elif seen:
        record('unit-exited', current=now, unit_state=current)
        raise SystemExit(0)
    elif time.monotonic() - started > 35:
        record('unit-not-started', current=now, unit_state=current)
        raise SystemExit(1)
    if tick and tick % 60 == 0:
        record('heartbeat', current=now, min_available_kib=min_available,
               min_swap_free_kib=min_swap_free, unit_state=current)
    tick += 1
    time.sleep(0.5)
