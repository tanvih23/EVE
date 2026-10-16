#watch a coordinated fleet solve one mission saves a path plot of the fleet to output/multiagent.png.
#  python -m experiments.multi_agent_demo
#  python -m experiments.multi_agent_demo --seed 7 --agents 4

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import Environment
from src.particle_filter import ParticleFilter
from src.mission_generator import random_mission
from src.multi_agent import MultiAgentMission
from src.visualize import plot_multiagent


def run(seed=7, n_agents=3):
    cfg = random_mission(seed, 'mixed', force_decoys=1)

    print(f"Mission seed {seed}  (1 decoy)\n")
    for n in (1, n_agents):
        np.random.seed(seed)
        env = Environment(cfg)
        pf = ParticleFilter(300, env.valid_cells, env.signal_noise,
                            signal_base=env.signal_base)
        rep = MultiAgentMission(env, pf, n_agents=n).run()
        status = (f"FOUND by unit {rep['found_by_unit']} at tick {rep['found_step']}"
                  if rep['found'] else "did not find EVE")
        print(f"  {n} unit(s): {status}  "
              f"| total actions={rep['total_actions']} "
              f"| final conf={rep['final_confidence']:.2f}")
        last = rep

    # plot the bigger fleet's run
    np.random.seed(seed)
    env = Environment(cfg)
    pf = ParticleFilter(300, env.valid_cells, env.signal_noise,
                        signal_base=env.signal_base)
    rep = MultiAgentMission(env, pf, n_agents=n_agents).run()
    path = plot_multiagent(rep, env, output_path='output/multiagent.png')
    if path:
        print(f"\nFleet path plot -> {path}")
    return rep


if __name__ == '__main__':
    seed, agents = 7, 3
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == '--seed' and i + 1 < len(args):
            seed = int(args[i + 1])
        if a == '--agents' and i + 1 < len(args):
            agents = int(args[i + 1])
    run(seed, agents)