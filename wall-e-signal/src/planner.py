#makes wall-e ask where he should go based on what hes alreasy discovered ie information gain

import numpy as np
from collections import deque


class EpistemicForager:
    """
    picks the action that teaches WALL-E the most about
    eve's location, adjusted for battery cost and strategy.
    """

    def __init__(self, env, particle_filter, safety_ratio=0.3,
                 exploit_threshold=0.7):
        #safety_ratio = fraction of battery to keep in reserve for getting
        #back to a charger
        #exploit_threshold = how sure we need to be before we stop exploring and head straight for the estimate
        self.env = env
        self.pf = particle_filter
        self.safety_ratio = safety_ratio
        self.exploit_threshold = exploit_threshold

        # cells we've been to to nudge exploration toward a new direction
        self.visited_cells = set()

        # log of which strategy was used each step for analysis
        self.strategy_log = []

        # signal readings kept for triangulation
        self.signal_readings = []

    def select_action(self, agent_pos, battery, max_battery, step_num):
        #picks the action weighing confidence battery info gain and the
        #safety margin returns action startegy and reasoning 
        confidence = self.pf.confidence()
        valid_actions = self.env.get_valid_actions(agent_pos, battery)

        if not valid_actions:
            return None, "stuck", "No valid actions available"

        battery_fraction = battery / max_battery

        # pick a strategy

        # 1) retreat- battery critically low
        if battery_fraction < self.safety_ratio:
            strategy = "retreat"
            action = self._retreat_action(agent_pos, valid_actions)
            reasoning = (f"Battery critical ({battery_fraction:.0%}). "
                        f"Heading to nearest recharge station.")

        # 2) exploit- confident enough about eve's position.
        # threshold allows for the roughening floor capping a converged
        # belief a bit below 1.0 too early = commits to decoys if too late it just wastes battery
        elif confidence > self.exploit_threshold:
            strategy = "exploit"
            action = self._exploit_action(agent_pos, valid_actions)
            est = self.pf.estimate()
            reasoning = (f"High confidence ({confidence:.1%}). "
                        f"Moving toward estimate ({est[0]:.0f}, {est[1]:.0f}).")

        # 3) explore- maximize information gain
        else:
            strategy = "explore"
            action, ig_score = self._explore_action(
                agent_pos, valid_actions, battery, max_battery
            )
            reasoning = (f"Exploring (conf={confidence:.1%}). "
                        f"Best IG action: {action} (score={ig_score:.4f}).")

        self.strategy_log.append({
            'step': step_num,
            'strategy': strategy,
            'confidence': float(confidence),
            'battery_fraction': float(battery_fraction),
            'action': action,
            'entropy': float(self.pf.grid_entropy()),
            'reasoning': reasoning,
        })

        return action, strategy, reasoning

    # strategy implementations

    def _explore_action(self, agent_pos, valid_actions, battery, max_battery):
        #pick the action with the best expected info gain per unit of battery
        current_entropy = self.pf.grid_entropy()

        best_action = None
        best_score = -float('inf')

        for action in valid_actions:
            if action == 'recharge':
                continue  # dont recharge while exploring

            cost = (self.env.battery_per_scan if action == 'scan'
                    else self.env.battery_per_move)

            ig = self._expected_info_gain(
                action, agent_pos, current_entropy
            )

            # score is the info gain per unit of battery
            score = ig / max(cost, 0.1)

            # small bonus for stepping onto a new cell
            if action in self.env.DIRECTIONS:
                new_pos = self._simulate_move(agent_pos, action)
                if new_pos not in self.visited_cells:
                    score *= 1.5  # 50% bonus for unexplored ground

            if score > best_score:
                best_score = score
                best_action = action

        return best_action or valid_actions[0], best_score

    def _expected_info_gain(self, action, agent_pos, current_entropy,
                            n_samples=20):
        #monte-carlo estimate of IG = H(now) - E[H(after)]. sample a few
        #guesses for where eve is, simulate the reading we'd get after this
        #action, see how much each one would shrink the entropy, average it.
        if action == 'scan':
            new_pos = agent_pos
        else:
            new_pos = self._simulate_move(agent_pos, action)

        # sample hypothetical eve positions from the current belief
        sample_indices = np.random.choice(
            self.pf.n_particles,
            size=min(n_samples, self.pf.n_particles),
            p=self.pf.weights,
            replace=True,
        )

        total_posterior_entropy = 0.0
        for idx in sample_indices:
            eve_hyp = tuple(self.pf.particles[idx].astype(int))
            # what signal would we see if eve were here
            sim_signal = self.env.signal_model_clean(new_pos, eve_hyp)
            # how would the belief change
            posterior = self.pf.hypothetical_update(
                sim_signal, new_pos, self.env.signal_model_clean
            )
            total_posterior_entropy += posterior.grid_entropy()

        expected_posterior = total_posterior_entropy / len(sample_indices)
        return current_entropy - expected_posterior

    def _exploit_action(self, agent_pos, valid_actions):
        #head straight for the estimated eve position
        target = self.pf.estimate()
        target_cell = (int(round(target[0])), int(round(target[1])))
        return self._move_toward(agent_pos, target_cell, valid_actions)

    def _retreat_action(self, agent_pos, valid_actions):
        #head for the nearest charger
        if 'recharge' in valid_actions:
            return 'recharge'

        # find nearest recharge station
        nearest = None
        min_dist = float('inf')
        for r in range(self.env.height):
            for c in range(self.env.width):
                if self.env.grid[r, c] == 2:
                    dist = abs(r - agent_pos[0]) + abs(c - agent_pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        nearest = (r, c)

        if nearest:
            return self._move_toward(agent_pos, nearest, valid_actions)

        # no charger found — just take any valid move
        move_actions = [a for a in valid_actions
                       if a in self.env.DIRECTIONS]
        return move_actions[0] if move_actions else valid_actions[0]

    # cross cheking particle filter

    def record_signal(self, pos, signal_strength):
        #stash a reading for triangulation later
        self.signal_readings.append({
            'pos': pos,
            'signal': signal_strength,
        })

    def triangulate(self):
        #independent estimate of eve using trilateration (same idea as GPS).
        #from signal = S0/(1+d^2) we get d = root(S0/signal - 1), so each
        #reading is a circle of radius d around where we took it, and 3+
        #circles intersect at eve. solved as linear least squares. returns
        #None with fewer than 3 usable readings.
        # keep readings with meaningful signal
        good = [r for r in self.signal_readings if r['signal'] > 0.01]
        if len(good) < 3:
            return None

        # use the 10 most recent readings
        recent = good[-10:]

        # turn signal strength into distance
        S0 = self.env.signal_base
        circles = []
        for reading in recent:
            s = reading['signal']
            if s >= S0:
                d = 0.0  # basically sitting? on eve
            else:
                d_sq = S0 / s - 1
                d = np.sqrt(max(0, d_sq))
            circles.append((np.array(reading['pos'], dtype=float), d))

        # linearize by subtracting the first circle from the rest
        r0, d0 = circles[0]
        A_rows = []
        b_rows = []
        for ri, di in circles[1:]:
            # (x-xi)^2+(y-yi)^2 = di^2 minus the same for circle 0
            # ie a linear equation in x,y
            A_rows.append(2 * (r0 - ri))
            b_rows.append(
                (di**2 - d0**2)
                - (np.sum(ri**2) - np.sum(r0**2))
            )

        if len(A_rows) < 1:
            return None

        A = np.array(A_rows)
        b = np.array(b_rows)

        # least squares handles the overdetermined case
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return result

    # utility methods

    def _move_toward(self, current, target, valid_actions):
        #head toward target along the shortest obstacle-free path. doing a quick BFS for
        #the real next step and only fall back to greedy if target is
        #unreachable
        step = self._bfs_first_step(current, target)
        if step is not None and step in valid_actions:
            return step
        return self._greedy_toward(current, target, valid_actions)

    def _bfs_first_step(self, start, goal):
        #first move on the shortest 4-connected path start->goal, or None
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        if start == goal:
            return None
        h, w, grid = self.env.height, self.env.width, self.env.grid
        goal = (min(max(goal[0], 0), h - 1), min(max(goal[1], 0), w - 1))
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for action, (dr, dc) in self.env.DIRECTIONS.items():
                nr, nc = cur[0] + dr, cur[1] + dc
                if (0 <= nr < h and 0 <= nc < w and grid[nr, nc] != 1
                        and (nr, nc) not in prev):
                    prev[(nr, nc)] = (cur, action)
                    q.append((nr, nc))
        if goal not in prev:
            return None
        # walk back from goal to the first step out of start
        node, action = goal, None
        while prev[node] is not None:
            parent, action = prev[node]
            node = parent
        return action

    def _greedy_toward(self, current, target, valid_actions):
        #fallback: whatever move cuts manhattan distance the most
        best_action, best_dist = None, float('inf')
        for action in valid_actions:
            if action not in self.env.DIRECTIONS:
                continue
            dr, dc = self.env.DIRECTIONS[action]
            dist = abs(current[0] + dr - target[0]) + abs(current[1] + dc - target[1])
            if dist < best_dist:
                best_dist, best_action = dist, action
        return best_action or valid_actions[0]

    def _simulate_move(self, pos, action):
        #where hed end up after a move without actually doing it
        if action not in self.env.DIRECTIONS:
            return pos
        dr, dc = self.env.DIRECTIONS[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        if (0 <= nr < self.env.height and 0 <= nc < self.env.width
                and self.env.grid[nr, nc] != 1):
            return (nr, nc)
        return pos

    def mark_visited(self, pos):
        #noting that weve visited a cell
        self.visited_cells.add(pos)