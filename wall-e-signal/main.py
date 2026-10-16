#the mission runner, the main entry point that runs a full mission from start to end,
#saves the trace n report and returns analysis plto
import sys
import os
import json

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment import Environment
from src.particle_filter import ParticleFilter
from src.planner import EpistemicForager
from src.agent import WallEAgent
from src.visualize import (
    ascii_map, print_mission_report, plot_mission_analysis
)


def run_mission(config_path, show_ascii=False, n_particles=500):
    #run one full mission end to end
    print(f"\n{'='*50}")
    print(f"  WALL-E & EVE's Signal — Mission Runner")
    print(f"  Config: {config_path}")
    print(f"  Particles: {n_particles}")
    print(f"{'='*50}\n")

    # 1) build the world
    print("Loading environment...")
    env = Environment(config_path)
    print(f"  Grid: {env.width}×{env.height}")
    print(f"  Obstacles: {(env.grid == 1).sum()}")
    print(f"  Recharge stations: {(env.grid == 2).sum()}")
    print(f"  Clues placed: {len(env.clues)}")
    print(f"  Decoys: {len(env.decoys)}")

    # 2) set up the belief tracker
    print("\nInitializing particle filter...")
    pf = ParticleFilter(
        n_particles=n_particles,
        valid_cells=env.valid_cells,
        signal_noise_std=env.signal_noise,
        signal_base=env.signal_base,
    )
    print(f"  Particles: {n_particles}")
    print(f"  Valid cells: {len(env.valid_cells)}")
    print(f"  Initial entropy: {pf.grid_entropy():.2f} bits")

    # 3) set up the planner
    print("\nInitializing epistemic forager...")
    planner = EpistemicForager(env, pf)

    # 4) build and run the agent
    print("\nCreating WALL-E agent...")
    agent = WallEAgent(env, pf, planner)

    print(f"  Start: {agent.pos}")
    print(f"  Battery: {agent.battery}")
    print(f"  Max steps: {agent.max_steps}")

    print("\n" + "─" * 50)
    print("  MISSION START")
    print("─" * 50 + "\n")

    # callback for the live print-out as wall-e moves
    def on_step(ag, action, strategy, observation):
        if show_ascii and ag.step_num % 10 == 0:
            print(ascii_map(env, ag.pos, pf, ag.step_num,
                          ag.battery, strategy))
        elif ag.step_num % 20 == 0:
            conf = pf.confidence()
            est = pf.estimate()
            print(f"  Step {ag.step_num:3d} │ "
                  f"Battery: {ag.battery:3d} │ "
                  f"Conf: {conf:.1%} │ "
                  f"Strategy: {strategy:8s} │ "
                  f"Est: ({est[0]:.1f}, {est[1]:.1f})")

        if observation.get('found_eve'):
            print(f"\n  ★ EVE FOUND at step {ag.step_num}! ★\n")
            if show_ascii:
                print(ascii_map(env, ag.pos, pf, ag.step_num,
                              ag.battery, 'FOUND'))

    report = agent.run_mission(on_step=on_step)

    # print the report
    print(print_mission_report(report))

    # 5) save outputs
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # save trace
    trace_path = agent.save_results(output_dir)
    print(f"\n  Trace saved to: {trace_path}")

    # save report
    report_path = os.path.join(output_dir, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to: {report_path}")

    # make the analysis plots
    print("\n  Generating analysis plots...")
    plot_path = plot_mission_analysis(
        agent.trace, report, env, pf,
        output_path=os.path.join(output_dir, 'analysis.png')
    )
    if plot_path:
        print(f"  Analysis plot saved to: {plot_path}")

    return report


if __name__ == '__main__':
    config = 'config/mission_1.yaml'
    show_ascii = False

    for arg in sys.argv[1:]:
        if arg == '--ascii':
            show_ascii = True
        elif arg.endswith('.yaml') or arg.endswith('.yml'):
            config = arg

    run_mission(config, show_ascii=show_ascii)