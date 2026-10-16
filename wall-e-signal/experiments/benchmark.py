#the generalization + robustness benchmark runs the agent over hundreds of random seeded missions and reports
#aggregate numbers, 3 studies: planner comparison, robustness sweep, multi-agent scaling
#everything is seeded so it reproduces exactly.
#  python -m experiments.benchmark            # full run
#  python -m experiments.benchmark --quick    # fewer missions, fast

import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import Environment
from src.particle_filter import ParticleFilter
from src.agent import WallEAgent
from src.planner import EpistemicForager
from src.baselines import make_planner, PLANNER_NAMES
from src.mission_generator import random_mission
from src.multi_agent import MultiAgentMission

OUT_DIR = os.path.join('output', 'benchmark')
N_PARTICLES = 300


def _new_filter(env):
    return ParticleFilter(N_PARTICLES, env.valid_cells, env.signal_noise,
                          signal_base=env.signal_base)


def _belief_error(report, env):
    #L1 distance between the final estimate and eve's true cell
    est = report['eve_location']['particle_filter_estimate']
    er, ec = env.eve_pos
    return abs(est[0] - er) + abs(est[1] - ec)


def _run_single(cfg, planner_name, seed):
    #run one planner on one mission, return a metrics dict
    np.random.seed(seed)
    env = Environment(cfg)
    pf = _new_filter(env)
    planner = make_planner(planner_name, env, pf, seed=seed)
    agent = WallEAgent(env, pf, planner)
    report = agent.run_mission()
    s = report['mission_summary']
    return {
        'found': s['outcome'] == 'EVE_FOUND',
        'outcome': s['outcome'],
        'steps': s['steps_taken'],
        'belief_error': _belief_error(report, env),
        'confidence': report['eve_location']['confidence'],
    }


# study 1- planner comparison

def benchmark_planners(difficulties, n_missions):
    print("\n[1/3] PLANNER COMPARISON")
    results = {}
    for diff in difficulties:
        results[diff] = {}
        print(f"  difficulty = {diff}")
        for name in PLANNER_NAMES:
            runs = []
            for i in range(n_missions):
                cfg = random_mission(1000 + i, diff)
                runs.append(_run_single(cfg, name, seed=1000 + i))
            found = [r for r in runs if r['found']]
            agg = {
                'success_rate': round(len(found) / len(runs), 3),
                'median_steps_to_find': (int(np.median([r['steps'] for r in found]))
                                         if found else None),
                'median_belief_error': round(float(np.median(
                    [r['belief_error'] for r in runs])), 2),
                'n_missions': len(runs),
            }
            results[diff][name] = agg
            print(f"    {name:14s} success={agg['success_rate']:.0%}  "
                  f"steps={agg['median_steps_to_find']}  "
                  f"belief_err={agg['median_belief_error']}")
    return results


# study 2- robustness sweeps

def robustness_sweep(n_missions):
    print("\n[2/3] ROBUSTNESS SWEEP (epistemic planner)")
    noise_levels = [0.4, 0.8, 1.2, 1.6, 2.0]
    decoy_counts = [0, 1, 2, 3, 4]

    noise_out = []
    print("  sensor-noise sweep (0 decoys):")
    for noise in noise_levels:
        runs = []
        for i in range(n_missions):
            cfg = random_mission(2000 + i, 'mixed',
                                 force_noise=noise, force_decoys=0)
            runs.append(_run_single(cfg, 'epistemic', seed=2000 + i))
        sr = np.mean([r['found'] for r in runs])
        be = float(np.median([r['belief_error'] for r in runs]))
        noise_out.append({'noise': noise, 'success_rate': round(float(sr), 3),
                          'median_belief_error': round(be, 2)})
        print(f"    noise={noise:.1f}  success={sr:.0%}  belief_err={be:.2f}")

    decoy_out = []
    print("  decoy-count sweep (noise=0.8):")
    for d in decoy_counts:
        runs = []
        for i in range(n_missions):
            cfg = random_mission(3000 + i, 'mixed',
                                 force_noise=0.8, force_decoys=d)
            runs.append(_run_single(cfg, 'epistemic', seed=3000 + i))
        sr = np.mean([r['found'] for r in runs])
        be = float(np.median([r['belief_error'] for r in runs]))
        decoy_out.append({'decoys': d, 'success_rate': round(float(sr), 3),
                          'median_belief_error': round(be, 2)})
        print(f"    decoys={d}  success={sr:.0%}  belief_err={be:.2f}")

    return {'noise_sweep': noise_out, 'decoy_sweep': decoy_out}


# study 3- multi-agent scaling

def multiagent_scaling(n_missions):
    print("\n[3/3] MULTI-AGENT SCALING (shared belief)")
    out = {}
    for n_agents in [1, 2, 3]:
        founds, steps = [], []
        for i in range(n_missions):
            cfg = random_mission(4000 + i, 'mixed', force_decoys=1)
            np.random.seed(4000 + i)
            env = Environment(cfg)
            pf = _new_filter(env)
            mission = MultiAgentMission(env, pf, n_agents=n_agents)
            r = mission.run()
            founds.append(r['found'])
            if r['found']:
                steps.append(r['found_step'])
        sr = float(np.mean(founds))
        out[n_agents] = {
            'success_rate': round(sr, 3),
            'median_ticks_to_find': (int(np.median(steps)) if steps else None),
            'n_missions': n_missions,
        }
        print(f"  {n_agents} agent(s): success={sr:.0%}  "
              f"ticks={out[n_agents]['median_ticks_to_find']}")
    return out


def main(quick=False):
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    n_cmp = 10 if quick else 20
    n_rob = 8 if quick else 15
    n_ma = 8 if quick else 15
    diffs = ['easy', 'mixed'] if quick else ['easy', 'mixed', 'hard']

    results = {
        'config': {'n_particles': N_PARTICLES, 'quick': quick,
                   'n_compare': n_cmp, 'n_robust': n_rob, 'n_multiagent': n_ma},
        'planner_comparison': benchmark_planners(diffs, n_cmp),
        'robustness': robustness_sweep(n_rob),
        'multiagent': multiagent_scaling(n_ma),
    }

    results['elapsed_seconds'] = round(time.time() - t0, 1)
    path = os.path.join(OUT_DIR, 'benchmark_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {results['elapsed_seconds']}s — results -> {path}")

    # plots skips if matplot missing
    try:
        from src.visualize import plot_benchmark
        plot_path = plot_benchmark(results, os.path.join(OUT_DIR, 'benchmark.png'))
        if plot_path:
            print(f"Charts -> {plot_path}")
    except Exception as e:
        print(f"(plotting skipped: {e})")

    return results


if __name__ == '__main__':
    main(quick='--quick' in sys.argv)