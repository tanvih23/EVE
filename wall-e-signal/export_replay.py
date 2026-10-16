#builds a bank of real recorded missions for the new map shuffle button.
#  python export_pool.py         - output/pool.json
#  python inject_replays.py      - bakes pool + the two missions into index.html
import numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.environment import Environment
from src.particle_filter import ParticleFilter
from src.planner import EpistemicForager
from src.agent import WallEAgent
from src.mission_generator import random_mission

N_WANT = 8          # how many maps in the bank
STEP_CAP = 90       # skip runs that drag on, keeps replays snappy
N_PARTICLES = 400


def run_once(cfg, seed):
    np.random.seed(seed)
    env = Environment(cfg)
    pf = ParticleFilter(N_PARTICLES, env.valid_cells, env.signal_noise,
                        signal_base=env.signal_base)
    agent = WallEAgent(env, pf, EpistemicForager(env, pf))
    steps = []
    prev = [None]
    ncells = max(2, len(env.valid_cells))
    log2n = float(np.log2(ncells))

    def on_step(ag, action, strategy, obs):
        dm = pf.density_map(env.height, env.width)
        flat = dm.flatten()
        tot = float(flat.sum()) or 1.0
        prob = flat / tot
        mx = float(dm.max()) or 1.0
        q = (np.clip(dm / mx, 0, 1) * 255).astype(int).flatten().tolist()
        est = pf.estimate()
        npar = pf.n_particles
        sidx = np.random.choice(npar, min(48, npar), replace=False)
        pts = [[int(pf.particles[i][0]), int(pf.particles[i][1])] for i in sidx]
        why = ag.planner.strategy_log[-1]['reasoning'] if ag.planner.strategy_log else ''
        pkp = float(prob.max())
        srt = np.sort(prob)[::-1]; cum = np.cumsum(srt)
        cr90 = int(np.searchsorted(cum, 0.90) + 1)
        entn = round(float(pf.grid_entropy()) / log2n, 3)
        p = prob + 1e-12; p /= p.sum()
        if prev[0] is None:
            ig = 0.0
        else:
            q0 = prev[0] + 1e-12; q0 /= q0.sum()
            ig = float(np.sum(p * np.log(p / q0)))
        prev[0] = prob.copy()
        modes = 0
        thr = 0.22 * mx
        for rr in range(env.height):
            for cc in range(env.width):
                v = dm[rr, cc]
                if v <= thr: continue
                loc = True
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < env.height and 0 <= nc < env.width and dm[nr, nc] > v:
                            loc = False; break
                    if not loc: break
                if loc: modes += 1
        steps.append({
            'pkp': round(pkp, 4), 'cr90': cr90, 'entn': entn,
            'ig': round(max(0.0, ig), 3), 'modes': int(modes),
            's': ag.step_num, 'p': [int(ag.pos[0]), int(ag.pos[1])],
            'b': int(ag.battery), 'c': round(float(pf.confidence()), 3),
            'e': round(float(pf.grid_entropy()), 2),
            'ne': round(float(pf.effective_sample_size()), 1),
            'st': strategy, 'a': action,
            'est': [round(float(est[0]), 1), round(float(est[1]), 1)],
            'cl': 1 if obs.get('clue') else 0,
            'sig': round(float(obs.get('signal', 0.0)), 2),
            'why': why, 'pt': pts, 'd': q,
        })

    report = agent.run_mission(on_step=on_step)
    return env, report, steps


def pack(env, report, steps, seed):
    meta = {
        'w': env.width, 'h': env.height,
        'obstacles': [[r, c] for r in range(env.height) for c in range(env.width)
                      if env.grid[r, c] == 1],
        'recharge': [[r, c] for r in range(env.height) for c in range(env.width)
                     if env.grid[r, c] == 2],
        'eve': [int(env.eve_pos[0]), int(env.eve_pos[1])],
        'decoys': [[int(d['position'][0]), int(d['position'][1])] for d in env.decoys],
        'clues': [[int(r), int(c)] for (r, c) in env.clues.keys()],
        'start': [int(env.config['agent']['start'][0]), int(env.config['agent']['start'][1])],
        'maxb': int(env.config['agent']['battery']),
        'maxs': int(env.config['time']['max_steps']),
        'outcome': report['mission_summary']['outcome'],
        'taken': report['mission_summary']['steps_taken'],
        'seed': seed, 'np': N_PARTICLES,
        'next': report['recommended_next_step'],
    }
    return {'meta': meta, 'steps': steps}


def main():
    os.makedirs('output', exist_ok=True)
    missions, used_eve, seed = [], set(), 0
    while len(missions) < N_WANT and seed < 400:
        cfg = random_mission(seed, 'easy')
        env, report, steps = run_once(cfg, seed)
        ev = tuple(int(x) for x in env.eve_pos)
        ok = (report['mission_summary']['outcome'] == 'EVE_FOUND'
              and len(steps) <= STEP_CAP and ev not in used_eve)
        if ok:
            used_eve.add(ev)
            missions.append(pack(env, report, steps, seed))
            m = missions[-1]['meta']
            print(f"  map {len(missions)}: seed={seed} grid={m['w']}x{m['h']} "
                  f"eve={m['eve']} decoys={len(m['decoys'])} steps={m['taken']}")
        seed += 1

    out = 'output/pool.json'
    with open(out, 'w') as f:
        json.dump({'missions': missions}, f, separators=(',', ':'))
    kb = os.path.getsize(out) / 1024
    print(f"\n{out}: {len(missions)} maps, size={kb:.0f}KB")


if __name__ == '__main__':
    main()