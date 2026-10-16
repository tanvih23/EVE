# finding eve

A little robot wakes up on a dead planet. Somewhere out there is EVE. He can't see her — all he gets is a faint, noisy signal that pings louder when he's closer, a few scattered clues, and a battery that won't last forever. There are even fake beacons planted around to throw him off. His job: figure out where EVE is and get to her before the power runs out.

That's the whole project. Under the cute framing it's a question I find genuinely interesting: **how should an agent act when it doesn't know where the thing it's looking for is, and every move costs something?** Not "walk toward the loudest beep" (that walks you straight into a decoy), but something closer to how a good searcher actually thinks — *where would a measurement teach me the most right now?*

That idea has a name: **epistemic foraging**. The agent forages for information, not just for the goal.

---

## how it actually works

Two pieces do the heavy lifting.

**1. A belief, not a guess.** WALL-E never commits to a single "EVE is probably there." Instead he keeps a cloud of a few hundred guesses — particles — spread over every cell she could be in. Every time he gets a signal reading, each particle gets reweighted by how well it explains what he just heard, then the cloud resamples toward the plausible spots. This is a **particle filter** (sequential Monte Carlo), and the nice thing is it handles the messiness honestly: noisy sensors, multiple competing hotspots, decoys that look real for a while. The belief is a full probability map, so I can measure how *confused* he still is at any moment — that's the entropy.

**2. Acting to learn.** Given that belief, where should he go? The planner scores each possible move by its **expected information gain** — roughly, how much it expects the belief to sharpen if it takes that move — and divides by what the move costs in battery. Early on, when he's lost, this pulls him toward measurements that split the map's uncertainty. Once the belief collapses onto one spot with enough confidence, he stops foraging and just goes and gets her. Obstacle-aware pathfinding (BFS) keeps him from walking into walls, and he'll detour to a charger if the battery's getting scary.

So the behaviour you see — spreading out, circling ambiguous regions, ignoring a loud-but-suspicious beacon, then suddenly committing — isn't scripted. It falls out of those two rules.

---

## what's in here

```
finding-eve/
├── main.py                  run one mission, watch it play out
├── requirements.txt
├── config/
│   ├── mission_1.yaml       the two hand-made worlds
│   └── mission_2.yaml
├── src/
│   ├── environment.py       the world: grid, signals, clues, decoys, battery
│   ├── particle_filter.py   the belief tracker
│   ├── planner.py           the epistemic forager (the brain)
│   ├── agent.py             glues belief + planner + world into a mission loop
│   ├── baselines.py         dumber planners to compare against
│   ├── mission_generator.py random solvable worlds for fair testing
│   ├── multi_agent.py       a coordinated fleet sharing one belief
│   └── visualize.py         plots + ascii map
├── experiments/
│   ├── benchmark.py         the "does this actually generalise" study
│   └── multi_agent_demo.py  1 robot vs a 3-robot team, head to head
├── export_replay.py         records a real mission → json (for the website)
├── export_pool.py           records a bank of random missions (the "new map" button)
├── inject_replays.py        bakes those recordings into index.html
├── index.html               the website (self-contained, no server needed)
└── BONUS.md                 notes on the extra credit stuff
```

---

## running the whole thing, start to finish

You need Python 3.10+. From the project folder:

```bash
# 1. set up
python -m venv venv
source venv/bin/activate          # windows: venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# 2. watch one mission play out
python main.py                    # runs mission 1
python main.py config/mission_2.yaml
python main.py --ascii            # same, but prints the map step by step in the terminal
```

This prints a full report — where he thought EVE was vs where she actually was, how many steps and how much battery it took, the final confidence — and saves an analysis plot to `output/`.

```bash
# 3. the real proof: does it generalise?
python -m experiments.benchmark          # full run (a few minutes)
python -m experiments.benchmark --quick  # faster, fewer worlds
```

One demo mission proves nothing — maybe he just got lucky on that map. So this runs the agent across hundreds of *random* worlds and stacks it against the baselines. It also sweeps sensor noise and decoy count to see where it breaks. Results land in `output/benchmark/`.

```bash
# 4. the team version
python -m experiments.multi_agent_demo
```

Same map solved once with one robot, once with three sharing a belief — prints the head-to-head.

That's the backend, end to end. **Everything above is the actual algorithm** — nothing's faked.

### the website part

The site (`index.html`) doesn't run Python live — the browser can't. Instead it *plays back real recordings* of the algorithm, baked right into the page. You only touch this step if you change the algorithm and want the website to reflect it:

```bash
python export_replay.py     # re-record the two feature missions
python export_pool.py        # re-record the random-map bank ("new map" button)
python inject_replays.py     # splice all of it into index.html
```

To see the site, just open `index.html` in a browser — double-click it, no server. To put it online, it's a single static file, so any host works (I used Vercel: `vercel --prod` from the folder).

---

## the extra-credit stuff, and what it taught me

I didn't just want the agent to work on an easy map, so I pushed on the hard parts (full notes in `BONUS.md`):

- **Decoys** — fake beacons planted away from EVE. A greedy "go to the loudest signal" agent gets suckered by these every time. The belief-based one stays skeptical because a decoy can't explain *all* the readings consistently. In the benchmark, decoys are clearly the hardest axis — that's where success rate drops first, and honestly that feels right.
- **Noisy sensors** — I cranked the signal noise up and watched it degrade gracefully rather than fall off a cliff; it holds up to roughly σ≈1.6 before really struggling.
- **A coordinated fleet** — three robots sharing one belief find EVE around 2.4× faster than one, and the trick that makes it work is a repulsion term so they fan out instead of all crowding the same loud spot.
- **Fair comparison** — against the baselines (random, greedy-to-signal, lawnmower sweep), the epistemic planner wins on success rate and finds EVE with far tighter localisation (around a fifth of a cell of error on the easy worlds vs the baselines flailing).

The honest version: it's strongest when the signal is informative and the map isn't wall-to-wall decoys. That's the frontier I'd keep pushing if I had more time.

---

## the website

The front end is its own thing — a single page that lets you actually *watch* a mission: WALL-E's path, his belief lighting up the grid cell by cell, the particle cloud, the 90% credible set tightening, and a full readout of why he made each decision. There's a "new map" button that drops you into a different real recorded solve, step / auto-run / scrub controls, and an in-depth analysis section underneath. It's all driven by the real trace data, not animation for show.

---

## why I built it this way

I could have made a flashier demo that cheated — hardcode the path, fake the "thinking." I didn't, because the interesting claim here is that *good search behaviour emerges from reasoning about uncertainty*, and that's only worth anything if the thing on screen is the thing actually running. So the benchmark is the heart of it, and the website is just an honest window into what the benchmark proves.

Built by Tanvi · for TAM-VIT.
