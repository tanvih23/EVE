# Bonus & Stretch Features

Everything below extends the core epistemic-foraging agent. All of it runs
in this repo with `numpy`, `pyyaml`, `matplotlib` — no simulators, no GPUs.

> **The brief's bonus list:** *visualize the path, clue locations and
> strategy changes (ASCII/UI/3D); decoy/spoofed signals; noisy sensors;
> multi-agent coordination; LLM vs. classical planner comparison;
> generalization benchmark across random missions; physics simulation
> (Gazebo/Webots).*

## What's implemented

| Bonus task | Status | Where |
|---|---|---|
| Visualize path / clues / strategy changes | ✅ | `src/visualize.py` (ASCII map + analysis plot) |
| Decoy / spoofed signals | ✅ | `src/environment.py`, robust likelihood in `src/particle_filter.py` |
| Noisy sensors | ✅ | Gaussian sensor model + a dedicated noise sweep |
| Multi-agent coordination | ✅ | `src/multi_agent.py` |
| LLM vs. classical planner comparison | ✅ (framework) | `src/baselines.py`, `experiments/benchmark.py` |
| Generalization benchmark across random missions | ✅ | `src/mission_generator.py`, `experiments/benchmark.py` |
| Physics simulation (Gazebo/Webots) | ⛔ out of scope here | see note below |

---

### 1. Visualization — path, clues, strategy changes
`plot_mission_analysis()` now marks **clue locations** (found vs still
hidden), **decoy sources**, the **start cell** and the full **path**, on
top of the belief heatmap. The strategy-timeline panel draws a dashed line
and label at every **strategy switch** (explore → exploit → retreat), so
you can see exactly when the agent stops hunting for information and
commits to an approach.

- `output/analysis.png` — clean success (mission 1, no decoys)
- `output/analysis_demo.png` — a 2-decoy mission the agent still solves
- ASCII live map: `python main.py config/mission_1.yaml --ascii`

### 2. Decoy / spoofed signals
Decoys are extra inverse-square sources the sensor **cannot** tell apart
from EVE. The naive fix fails: because decoys only ever *add* signal, a
reading inflated by a nearby decoy wrongly kills particles at the true
(far, weak) EVE location, so the belief gets captured by the decoy.

The particle filter uses a **robust, asymmetric likelihood**
(`update_signal`): it tolerates "observed > predicted" (could be a decoy)
far more than "observed < predicted" (which additive decoys can't
explain), and clips any single residual so one bad reading can't annihilate
a hypothesis. Over many vantage points the one source consistent with
*all* readings — EVE — still wins.

### 3. Noisy sensors
Every reading is `signal + N(0, σ²)`. The benchmark sweeps σ from 0.4 to
2.0. Result: **100% success up to σ = 1.2, degrading gracefully to ~90% at
σ = 2.0**, with sub-cell localisation error throughout — the particle
filter averages noise out across readings.

### 4. Multi-agent coordination
`MultiAgentMission` runs a fleet of WALL-E units that **share one belief**
(centralised sensor fusion): every beep every unit hears sharpens the same
estimate. Coordination comes from a **repulsion term** — each unit's
information-gain score is penalised for heading toward a teammate, so the
fleet fans out instead of clumping. Measured payoff: **3 units find EVE in
~27 ticks vs ~79 for a single unit** (`experiments/multi_agent_demo.py`,
`output/multiagent.png`).

### 5. LLM vs. classical planner comparison
`src/baselines.py` implements four planners behind the same interface as
the epistemic forager, so any of them drops into the agent unchanged:

- `random` — floor baseline
- `lawnmower` — exhaustive coverage sweep (classic)
- `greedy` — stochastic hill-climb on the raw signal (the "obvious" method)
- `belief_greedy` — a **belief-summary policy**: it receives the same
  compact state description you'd put in a prompt (top candidate cells,
  confidence, battery) and acts on it. `LLMPlanner` is the same class with
  a pluggable `policy_fn` — **point it at an Anthropic API call for a
  genuine LLM policy**; with no key it falls back to the heuristic so the
  benchmark always runs offline. The "LLM slot" is explicit and swappable
  rather than faked.

The epistemic info-gain planner is the only one that reasons about
*expected uncertainty reduction*, and it wins across the board (numbers
below).

### 6. Generalization benchmark across random missions
`src/mission_generator.py` procedurally builds randomized, **reachability-
checked** missions (BFS guarantees EVE is always reachable). `experiments/
benchmark.py` runs three studies over hundreds of seeded missions and
writes `output/benchmark/benchmark_results.json` + `benchmark.png`:

1. planner comparison across difficulties,
2. robustness sweeps (noise, decoy count),
3. multi-agent scaling (1 / 2 / 3 units).

**Representative results** (success rate; lower belief-error is better):

| Planner | Easy | Mixed |
|---|---|---|
| **epistemic** | **~70–85%** | **~40%** |
| belief_greedy ("LLM-style") | ~40% | ~0–10% |
| greedy hill-climb | ~20% | ~10% |
| lawnmower | ~20% | ~10% |
| random | ~0–20% | ~0% |

Epistemic localises to **~0.2-cell** median error on fair missions vs
1.6–5.8 for the baselines. Decoys are the hard frontier for everyone —
quantified in the decoy sweep, and called out honestly as future work
(a mixture / multi-hypothesis sensor model would handle multiple sources
explicitly).

Run it:
```bash
python -m experiments.benchmark            # full (~6–8 min)
python -m experiments.benchmark --quick    # ~3 min
python -m experiments.multi_agent_demo     # fleet demo + plot
```

### 7. Physics simulation (Gazebo/Webots) — not included
Deliberately out of scope for this submission: Gazebo/Webots need a full
robotics stack (ROS, a display server, a physics engine) that doesn't
belong in a self-contained `numpy` repo, and the grid world already
captures the decision-making problem the project is about (belief under
noisy/ambiguous sensing). The clean seam for it is `Environment.step()` —
swap the grid dynamics for a simulator bridge and the particle filter,
planner, baselines and benchmark all run unchanged.

---

## Engine improvements made along the way
These were needed to make the bonus work correct and the benchmarks
tractable; they also improve the core agent:

- **Vectorized likelihood + entropy** — `grid_entropy` and the signal
  update are now array ops instead of Python loops: ~40× faster, which is
  what makes hundreds of benchmark missions feasible.
- **Regularized resampling (roughening)** — adds spread-scaled jitter after
  resampling so the filter can't freeze onto a single wrong cell
  ("sample impoverishment"), which is what made naive multi-agent fusion
  collapse.
- **BFS navigation** — `exploit`/`retreat` now follow the shortest
  *obstacle-free* path instead of greedy Manhattan steps, so the agent
  stops getting wall-stuck next to EVE. This alone took the easy-mission
  success rate from ~65% to ~85%.
