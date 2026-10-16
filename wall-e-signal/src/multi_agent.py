import numpy as np

from src.planner import EpistemicForager

class CoordinatedForager(EpistemicForager):
    #an epistemic forager that also avoids crowding its teammates=

    def __init__(self, env, particle_filter, unit_id, get_teammates,
                 repulsion=0.6, safety_ratio=0.3):
        super().__init__(env, particle_filter, safety_ratio)
        self.unit_id = unit_id
        self.get_teammates = get_teammates
        self.repulsion = repulsion

    def _explore_action(self, agent_pos, valid_actions, battery, max_battery):
        current_entropy = self.pf.grid_entropy()
        mates = self.get_teammates(self.unit_id)

        best_action, best_score = None, -float('inf')
        for action in valid_actions:
            if action == 'recharge':
                continue
            cost = (self.env.battery_per_scan if action == 'scan'
                    else self.env.battery_per_move)
            ig = self._expected_info_gain(action, agent_pos, current_entropy)
            score = ig / max(cost, 0.1)

            if action in self.env.DIRECTIONS:
                nxt = self._simulate_move(agent_pos, action)
                if nxt not in self.visited_cells:
                    score *= 1.5
                # crowding penalty distance to nearest teammate in [0,1]
                if mates:
                    dmin = min(abs(nxt[0] - m[0]) + abs(nxt[1] - m[1]) for m in mates)
                    crowd = np.exp(-dmin / 4.0)          # close mate -> ~1
                    score *= max(0.05, 1.0 - self.repulsion * crowd)

            if score > best_score:
                best_score, best_action = score, action

        return best_action or valid_actions[0], best_score


class MultiAgentMission:
    #run n_agents coordinated units on a shared belief until one reaches eve,
    #everyone's out of battery, or max_steps runs out
    # 

    def __init__(self, env, particle_filter, n_agents=3, repulsion=0.6,
                 starts=None):
        self.env = env
        self.pf = particle_filter
        self.n_agents = n_agents
        self.max_steps = env.config['time']['max_steps']

        base_start = tuple(env.config['agent']['start'])
        max_batt = env.config['agent']['battery']
        if starts is None:
            starts = self._spread_starts(base_start, n_agents)

        self.units = []
        for i in range(n_agents):
            planner = CoordinatedForager(
                env, particle_filter, unit_id=i,
                get_teammates=self._teammates_of, repulsion=repulsion,
            )
            self.units.append({
                'id': i,
                'pos': starts[i],
                'battery': max_batt,
                'max_battery': max_batt,
                'planner': planner,
                'path': [list(starts[i])],
                'alive': True,
            })
        for u in self.units:
            u['planner'].mark_visited(u['pos'])

        self.found_by = None
        self.found_step = None
        self.total_actions = 0

    def _spread_starts(self, base, n):
        #put units at distinct nearby valid cells -corners-ish
        h, w = self.env.height, self.env.width
        candidates = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1),
                      (0, w // 2), (h - 1, w // 2)]
        out = []
        for pos in candidates:
            if self.env.grid[pos] != 1 and pos not in out:
                out.append(pos)
            if len(out) == n:
                break
        while len(out) < n:
            out.append(base)
        return out

    def _teammates_of(self, uid):
        return [u['pos'] for u in self.units if u['id'] != uid and u['alive']]

    def run(self):
        init_sig = self.env.get_initial_signal()
        if init_sig:
            self.pf.update_initial_signal(init_sig, self.units[0]['pos'])

        for step in range(self.max_steps):
            any_alive = False
            for u in self.units:
                if not u['alive']:
                    continue
                if u['battery'] <= 0:
                    u['alive'] = False
                    continue
                any_alive = True

                action, strategy, _ = u['planner'].select_action(
                    u['pos'], u['battery'], u['max_battery'], step
                )
                if action is None:
                    continue

                new_pos, obs, cost = self.env.step(u['pos'], action)
                if action == 'recharge' and obs.get('recharged'):
                    u['battery'] = min(u['max_battery'], u['battery'] + obs['amount'])
                else:
                    u['battery'] -= cost

                # shared-belief fusion- every unit's reading updates the one
                # common filter
                if 'signal' in obs:
                    self.pf.update_signal(obs['signal'], new_pos,
                                          self.env.signal_model_clean)
                    u['planner'].record_signal(new_pos, obs['signal'])
                if 'clue' in obs:
                    self.pf.update_clue(obs['clue'], new_pos)

                u['planner'].mark_visited(new_pos)
                u['pos'] = new_pos
                u['path'].append(list(new_pos))
                self.total_actions += 1

                if obs.get('found_eve'):
                    self.found_by = u['id']
                    self.found_step = step
                    return self._report()

            if not any_alive:
                break

        return self._report()

    def _report(self):
        est = self.pf.estimate()
        return {
            'n_agents': self.n_agents,
            'found': self.found_by is not None,
            'found_by_unit': self.found_by,
            'found_step': self.found_step,
            'total_actions': self.total_actions,
            'final_estimate': [round(float(est[0]), 2), round(float(est[1]), 2)],
            'final_confidence': round(float(self.pf.confidence()), 4),
            'final_entropy': round(float(self.pf.grid_entropy()), 2),
            'paths': [u['path'] for u in self.units],
            'unit_batteries': [u['battery'] for u in self.units],
        }