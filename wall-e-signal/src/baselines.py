#other planners to compare the epistemic forager against to prove info gain actually is better than 
#other approaches
#all these planners expose the same interface the agent calls (select_action,
#mark_visited, record_signal, triangulate + the visited/signal/strategy lists)
#so any of them drops straight into WallEAgent.


import numpy as np


class BasePlanner:
    #shared bookkeeping so every baseline satisfies the agent interface

    def __init__(self, env, seed=0):
        self.env = env
        self.rng = np.random.RandomState(seed)
        self.visited_cells = set()
        self.signal_readings = []
        self.strategy_log = []

    # interface the agent relies on
    def mark_visited(self, pos):
        self.visited_cells.add(pos)

    def record_signal(self, pos, signal_strength):
        self.signal_readings.append({'pos': pos, 'signal': signal_strength})

    def triangulate(self):
        # baselines dont triangulate; the report just leaves it out
        return None

    def _log(self, step_num, strategy, action, reasoning):
        self.strategy_log.append({
            'step': step_num,
            'strategy': strategy,
            'action': action,
            'reasoning': reasoning,
        })

    # helpers
    def _move_toward(self, current, target, valid_actions):
        best, best_d = None, float('inf')
        for a in valid_actions:
            if a not in self.env.DIRECTIONS:
                continue
            dr, dc = self.env.DIRECTIONS[a]
            d = abs(current[0] + dr - target[0]) + abs(current[1] + dc - target[1])
            if d < best_d:
                best_d, best = d, a
        return best

    def _simulate_move(self, pos, action):
        if action not in self.env.DIRECTIONS:
            return pos
        dr, dc = self.env.DIRECTIONS[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        if (0 <= nr < self.env.height and 0 <= nc < self.env.width
                and self.env.grid[nr, nc] != 1):
            return (nr, nc)
        return pos


class RandomPlanner(BasePlanner):
    #floor baseline: any legal action, biased toward moving

    def select_action(self, agent_pos, battery, max_battery, step_num):
        valid = self.env.get_valid_actions(agent_pos, battery)
        if not valid:
            return None, 'stuck', 'no legal actions'
        moves = [a for a in valid if a in self.env.DIRECTIONS]
        # 85% move, 15% scan/other — just wandering
        if moves and self.rng.random() < 0.85:
            action = moves[self.rng.randint(len(moves))]
        else:
            action = valid[self.rng.randint(len(valid))]
        self._log(step_num, 'random', action, 'random legal action')
        return action, 'random', 'random legal action'


class LawnmowerPlanner(BasePlanner):
    #systematic coverage- sweep east/west row by row, stepping south at the end
    #of each row guaranteed to find eve eventually but completely ignores the
    #signal so it wastes the whole sensor.

    def __init__(self, env, seed=0):
        super().__init__(env, seed)
        self._dir = 'east'

    def select_action(self, agent_pos, battery, max_battery, step_num):
        valid = self.env.get_valid_actions(agent_pos, battery)
        if not valid:
            return None, 'stuck', 'no legal actions'

        # top up if we're low and happen to be on a charger
        if battery / max_battery < 0.25 and 'recharge' in valid:
            self._log(step_num, 'coverage', 'recharge', 'topping up battery')
            return 'recharge', 'coverage', 'topping up battery'

        horiz = self._dir
        other = 'west' if horiz == 'east' else 'east'

        # keep going along the current sweep direction
        if horiz in valid and self._simulate_move(agent_pos, horiz) != agent_pos:
            action = horiz
        elif 'south' in valid and self._simulate_move(agent_pos, 'south') != agent_pos:
            action = 'south'
            self._dir = other  # flip horizontal direction on the next row
        elif other in valid and self._simulate_move(agent_pos, other) != agent_pos:
            action = other
        else:
            moves = [a for a in valid if a in self.env.DIRECTIONS]
            action = moves[self.rng.randint(len(moves))] if moves else valid[0]

        self._log(step_num, 'coverage', action, f'sweep ({self._dir})')
        return action, 'coverage', f'sweep ({self._dir})'


class GreedySignalPlanner(BasePlanner):
    #remembers the signal at every cell we've stood on and walk towards, random step when there's no
    #uphill move. it takes the signal at face value, so a decoy thats closer
    #than eve captures it for good.

    def __init__(self, env, seed=0):
        super().__init__(env, seed)
        self.signal_memory = {} 

    def record_signal(self, pos, signal_strength):
        super().record_signal(pos, signal_strength)
        self.signal_memory[pos] = signal_strength

    def select_action(self, agent_pos, battery, max_battery, step_num):
        valid = self.env.get_valid_actions(agent_pos, battery)
        if not valid:
            return None, 'stuck', 'no legal actions'

        if battery / max_battery < 0.25 and 'recharge' in valid:
            self._log(step_num, 'greedy', 'recharge', 'battery low')
            return 'recharge', 'greedy', 'battery low'

        here = self.signal_memory.get(agent_pos, -1.0)
        best_action, best_sig = None, here
        for a in valid:
            if a not in self.env.DIRECTIONS:
                continue
            nxt = self._simulate_move(agent_pos, a)
            s = self.signal_memory.get(nxt, None)
            if s is not None and s > best_sig:
                best_sig, best_action = s, a

        if best_action is not None:
            action, why = best_action, f'uphill toward signal {best_sig:.2f}'
        else:
            # no known uphill neighbour so will either probe somewhere new or else wander
            unexplored = [a for a in valid if a in self.env.DIRECTIONS
                          and self._simulate_move(agent_pos, a) not in self.visited_cells]
            pool = unexplored or [a for a in valid if a in self.env.DIRECTIONS]
            action = pool[self.rng.randint(len(pool))] if pool else valid[0]
            why = 'probing (no uphill move)'

        self._log(step_num, 'greedy', action, why)
        return action, 'greedy', why


class BeliefGreedyPlanner(BasePlanner):
    #it gets a short summary of the belief and acts on it with a fixed sensible
    #policy instead of computing info gain
    #it never quantifies how much an action would reduce uncertainty,
    #which is the one thing the epistemic forager adds

    def __init__(self, env, particle_filter, seed=0):
        super().__init__(env, seed)
        self.pf = particle_filter
        self._last_scan_step = -10

    def belief_summary(self, agent_pos, battery, max_battery):
        #the state description a prompt based policy would get
        top = self.pf.top_cells(3)
        est = self.pf.estimate()
        return {
            'agent_pos': list(agent_pos),
            'battery_fraction': round(battery / max_battery, 2),
            'confidence': round(float(self.pf.confidence()), 3),
            'entropy_bits': round(float(self.pf.grid_entropy()), 2),
            'estimate': [round(float(est[0]), 1), round(float(est[1]), 1)],
            'top_cells': [{'cell': list(c), 'p': round(float(p), 3)} for c, p in top],
        }

    def _policy(self, summary, valid_actions, agent_pos, step_num):
        #builtin heuristic policy. llmplanner overrides this.
        if summary['battery_fraction'] < 0.25 and 'recharge' in valid_actions:
            return 'recharge', 'exploit', 'battery low -> recharge'
        if summary['confidence'] > 0.6:
            tgt = (int(round(summary['estimate'][0])), int(round(summary['estimate'][1])))
            a = self._move_toward(agent_pos, tgt, valid_actions)
            if a:
                return a, 'exploit', f"confident -> approach {tgt}"
        if 'scan' in valid_actions and step_num - self._last_scan_step >= 4:
            self._last_scan_step = step_num
            return 'scan', 'exploit', 'uncertain -> scan'
        tgt = (int(round(summary['estimate'][0])), int(round(summary['estimate'][1])))
        a = self._move_toward(agent_pos, tgt, valid_actions)
        if a:
            return a, 'exploit', 'drift toward belief centre'
        moves = [x for x in valid_actions if x in self.env.DIRECTIONS]
        return (moves[self.rng.randint(len(moves))] if moves else valid_actions[0],
                'exploit', 'fallback move')

    def select_action(self, agent_pos, battery, max_battery, step_num):
        valid = self.env.get_valid_actions(agent_pos, battery)
        if not valid:
            return None, 'stuck', 'no legal actions'
        summary = self.belief_summary(agent_pos, battery, max_battery)
        action, strategy, reasoning = self._policy(summary, valid, agent_pos, step_num)
        if action not in valid:
            action = valid[0]
        self._log(step_num, strategy, action, reasoning)
        return action, strategy, reasoning


class LLMPlanner(BeliefGreedyPlanner):

    def __init__(self, env, particle_filter, policy_fn=None, seed=0):
        super().__init__(env, particle_filter, seed)
        self.policy_fn = policy_fn

    def _policy(self, summary, valid_actions, agent_pos, step_num):
        if self.policy_fn is None:
            return super()._policy(summary, valid_actions, agent_pos, step_num)
        try:
            action = self.policy_fn(summary, valid_actions)
        except Exception as e:
            return super()._policy(summary, valid_actions, agent_pos, step_num)
        if action not in valid_actions:
            return super()._policy(summary, valid_actions, agent_pos, step_num)
        return action, 'llm', 'policy_fn choice'


# factory used by the benchmark harness
def make_planner(name, env, particle_filter, seed=0, policy_fn=None):
    #name -planner instance wired to this env/filter
    from src.planner import EpistemicForager
    name = name.lower()
    if name == 'epistemic':
        return EpistemicForager(env, particle_filter)
    if name == 'random':
        return RandomPlanner(env, seed)
    if name == 'lawnmower':
        return LawnmowerPlanner(env, seed)
    if name == 'greedy':
        return GreedySignalPlanner(env, seed)
    if name == 'belief_greedy':
        return BeliefGreedyPlanner(env, particle_filter, seed)
    if name == 'llm':
        return LLMPlanner(env, particle_filter, policy_fn, seed)
    raise ValueError(f'unknown planner: {name}')


PLANNER_NAMES = ['epistemic', 'belief_greedy', 'greedy', 'lawnmower', 'random']