#generates random worlds for the generalization benchmark
#so we make hundreds of random worlds and run on all of them
#each mission varies grid size, obstacle density, eve's spot + strength, number
#and placement of decoys, sensor noise, clue count and chargers.
#every mission we return is solvable because a bfs confirms eve is reachable from the start

import numpy as np
from collections import deque


def _reachable(grid, start, goal):
    #bfs over 4 non connected obstacles
    h, w = grid.shape
    if grid[start] == 1 or grid[goal] == 1:
        return False
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w
                    and grid[nr, nc] != 1 and (nr, nc) not in seen):
                seen.add((nr, nc))
                q.append((nr, nc))
    return False


def random_mission(seed, difficulty='mixed', force_noise=None,
                   force_decoys=None):
    #build one random mission config
    rng = np.random.RandomState(seed)

    if difficulty == 'easy':
        w, h = rng.randint(14, 20), rng.randint(10, 15)
        obs = rng.uniform(0.05, 0.12)
        n_decoys = rng.randint(0, 2)
        noise = rng.uniform(0.3, 0.8)
    elif difficulty == 'hard':
        w, h = rng.randint(24, 32), rng.randint(18, 24)
        obs = rng.uniform(0.15, 0.24)
        n_decoys = rng.randint(2, 5)
        noise = rng.uniform(1.0, 1.8)
    else:  # mixed
        w, h = rng.randint(18, 26), rng.randint(14, 20)
        obs = rng.uniform(0.10, 0.18)
        n_decoys = rng.randint(1, 4)
        noise = rng.uniform(0.6, 1.3)

    if force_noise is not None:
        noise = float(force_noise)
    if force_decoys is not None:
        n_decoys = int(force_decoys)

    start = (0, 0)
    # put eve in the far half so the mission isnt trivial
    eve = (rng.randint(h // 2, h), rng.randint(w // 2, w))
    while eve == start:
        eve = (rng.randint(h // 2, h), rng.randint(w // 2, w))

    # 2 3 chargers scattered around, never on start ot eve
    n_recharge = rng.randint(2, 4)
    recharge = []
    while len(recharge) < n_recharge:
        pos = [int(rng.randint(0, h)), int(rng.randint(0, w))]
        if tuple(pos) not in (start, eve) and pos not in recharge:
            recharge.append(pos)

    eve_strength = float(round(rng.uniform(6.0, 10.0), 1))
    decoys = []
    tries = 0
    while len(decoys) < n_decoys and tries < n_decoys * 40 + 40:
        tries += 1
        dr = int(rng.randint(0, h))
        dc = int(rng.randint(0, w))
        if (dr, dc) in (start, eve):
            continue
        # keep decoys a little away from eve so they genuinely mislead
        if abs(dr - eve[0]) + abs(dc - eve[1]) < 3:
            continue
        if any(d['position'] == [dr, dc] for d in decoys):
            continue
        decoys.append({
            'position': [dr, dc],
            'strength': float(round(rng.uniform(3.0, eve_strength * 0.9), 1)),
        })

    # 
    grid = _build_obstacles(w, h, obs, start, eve, recharge, decoys, rng)

    # 
    ang = np.degrees(np.arctan2(eve[1] - start[1], eve[0] - start[0]))
    ang += rng.normal(0, 20)

    battery = int(rng.randint(120, 200))
    max_steps = int(rng.randint(200, 360))

    return {
        'grid': {
            'width': int(w), 'height': int(h),
            'seed': int(seed),
            'obstacle_density': 0.0,         # we inject the grid directly
            '_prebuilt_obstacles': grid.tolist(),
            'recharge_stations': recharge,
        },
        'eve': {
            'position': [int(eve[0]), int(eve[1])],
            'status': 'dormant',
            'signal_strength': eve_strength,
        },
        'initial_signal': {
            'direction': float(round(ang, 1)),
            'strength': float(round(rng.uniform(0.1, 0.4), 2)),
            'timestamp': 0,
        },
        'clues': {'seed': int(seed), 'count': int(rng.randint(4, 9))},
        'decoys': decoys,
        'agent': {
            'start': [int(start[0]), int(start[1])],
            'battery': battery,
            'battery_per_move': 1,
            'battery_per_scan': int(rng.randint(3, 5)),
            'battery_recharge': int(rng.randint(30, 45)),
            'scan_radius': int(rng.randint(2, 4)),
            'signal_noise_std': float(round(noise, 2)),
        },
        'time': {'max_steps': max_steps},
        '_meta': {'difficulty': difficulty, 'n_decoys': len(decoys)},
    }


def _build_obstacles(w, h, density, start, eve, recharge, decoys, rng,
                     max_tries=40):
    #place obstacles but keep eve reachable from start
    protected = {start, eve} | {tuple(p) for p in recharge} \
        | {tuple(d['position']) for d in decoys}
    n = int(w * h * density)

    for attempt in range(max_tries):
        grid = np.zeros((h, w), dtype=int)
        placed = 0
        guard = 0
        while placed < n and guard < n * 30:
            guard += 1
            r, c = rng.randint(0, h), rng.randint(0, w)
            if (r, c) in protected or grid[r, c] == 1:
                continue
            grid[r, c] = 1
            placed += 1
        if _reachable(grid, start, eve):
            return grid
        density *= 0.8  # thin out and retry
        n = int(w * h * density)

    # fallback
    return np.zeros((h, w), dtype=int)