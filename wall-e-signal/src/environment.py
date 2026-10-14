#basically the 3d world wall-e explores including the charging stations, the clues eve leave
#for him etc.
import numpy as np
import yaml


class Environment:
    
    #the simulated world that wall-e explores loaded from the yaml files

    def __init__(self, config):
        if isinstance(config, dict):
            self.config = config
        else:
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)

        grid_cfg = self.config['grid']
        self.width = grid_cfg['width']
        self.height = grid_cfg['height']
        self.rng = np.random.RandomState(grid_cfg.get('seed', 42))

        # 0=empty 1=obstacle 2=recharge
        self.grid = np.zeros((self.height, self.width), dtype=int)
        prebuilt = grid_cfg.get('_prebuilt_obstacles', None)
        if prebuilt is not None:
            self.grid = np.array(prebuilt, dtype=int)
        else:
            self._place_obstacles(grid_cfg.get('obstacle_density', 0.15))
        self._place_recharge_stations(grid_cfg.get('recharge_stations', []))

        #the true state eve is in that the agent will never read directly
        eve_cfg = self.config['eve']
        self.eve_pos = tuple(eve_cfg['position'])
        self.eve_status = eve_cfg.get('status', 'dormant')
        self.signal_base = eve_cfg.get('signal_strength', 10.0)

        #decoy signal sources
        self.decoys = []
        for d in self.config.get('decoys', []):
            self.decoys.append({
                'position': tuple(d['position']),
                'strength': d['strength'],
            })

        # placing clues
        self._place_clues()

        # config of agent
        agent_cfg = self.config['agent']
        self.scan_radius = agent_cfg.get('scan_radius', 3)
        self.signal_noise = agent_cfg.get('signal_noise_std', 0.5)
        self.battery_per_move = agent_cfg.get('battery_per_move', 1)
        self.battery_per_scan = agent_cfg.get('battery_per_scan', 3)
        self.battery_recharge_amount = agent_cfg.get('battery_recharge', 30)

        # all the valid/non obstacle cell that the init of particlefilter uses
        self.valid_cells = [
            (r, c) for r in range(self.height)
            for c in range(self.width)
            if self.grid[r, c] != 1
        ]

        # sesnor rng, so that obstagle rng doesnt affect the signals
        self.sensor_rng = np.random.RandomState(
            grid_cfg.get('seed', 42) + 1000
        )

    # grid construction

    def _place_obstacles(self, density):
        #randomly placing obstacles avoiding start position and eve
        n = int(self.width * self.height * density)
        start = tuple(self.config['agent']['start'])
        eve = tuple(self.config['eve']['position'])

        placed = 0
        while placed < n:
            r = self.rng.randint(0, self.height)
            c = self.rng.randint(0, self.width)
            if (r, c) != start and (r, c) != eve and self.grid[r, c] == 0:
                self.grid[r, c] = 1
                placed += 1

    def _place_recharge_stations(self, positions):
        #to place the recharge stations so wall-e can recharge at specified positions
        for pos in positions:
            r, c = int(pos[0]), int(pos[1])
            if 0 <= r < self.height and 0 <= c < self.width:
                self.grid[r, c] = 2

    def _place_clues(self):
        #each clue has a position, a type, a direction with noise and a strength
        self.clues = {}
        clue_cfg = self.config.get('clues', {})
        n_clues = clue_cfg.get('count', 5)
        clue_rng = np.random.RandomState(clue_cfg.get('seed', 123))

        if 'positions' in clue_cfg:

            for pos_info in clue_cfg['positions']:
                r, c = pos_info['position']
                self.clues[(r, c)] = {
                    'type': pos_info.get('type', 'trace'),
                    'direction_hint': pos_info.get('direction', None),
                    'strength': pos_info.get('strength', 1.0),
                    'found': False,
                }
        else:
            # tp generate clues at random positions
            eve_r, eve_c = self.eve_pos
            placed = 0
            attempts = 0
            while placed < n_clues and attempts < n_clues * 20:
                attempts += 1
                # random positions
                cr = clue_rng.randint(0, self.height)
                cc = clue_rng.randint(0, self.width)

                if self.grid[cr, cc] != 0 or (cr, cc) in self.clues:
                    continue

                # direction clues towards eve but have noise
                dx = eve_c - cc
                dy = eve_r - cr
                angle_to_eve = np.degrees(np.arctan2(dy, dx))
                noisy_angle = angle_to_eve + clue_rng.normal(0, 25)

                dist = np.sqrt(dx**2 + dy**2)
                self.clues[(cr, cc)] = {
                    'type': clue_rng.choice(['trace', 'signal_echo', 'debris']),
                    'direction_hint': noisy_angle,
                    'strength': 1.0 / (1 + dist * 0.1),
                    'found': False,
                }
                placed += 1

    #signal physics

    def signal_at(self, pos):
        #basically combines eve's sig, the decoys and the noise and the particle filter tells
        #which is a decoy or which is real
        total_clean = self._eve_signal_clean(pos)

        # add decoy signals
        for decoy in self.decoys:
            dr, dc = decoy['position']
            dist_sq = (pos[0] - dr)**2 + (pos[1] - dc)**2
            total_clean += decoy['strength'] / (1.0 + dist_sq)

        # add noise
        noise = self.sensor_rng.normal(0, self.signal_noise)
        return max(0.0, total_clean + noise)

    def _eve_signal_clean(self, pos):
        #only eve's clean signal
        r, c = pos
        er, ec = self.eve_pos
        dist_sq = (r - er)**2 + (c - ec)**2
        return self.signal_base / (1.0 + dist_sq)

    def signal_model_clean(self, agent_pos, eve_hypothesis):

        #what the expected signal would be if eve was at a position
        ar, ac = agent_pos
        er, ec = int(eve_hypothesis[0]), int(eve_hypothesis[1])
        dist_sq = (ar - er)**2 + (ac - ec)**2
        return self.signal_base / (1.0 + dist_sq)

    # agent interaction

    DIRECTIONS = {
        'north': (-1, 0), 'south': (1, 0),
        'east': (0, 1), 'west': (0, -1),
    }

    def step(self, agent_pos, action):
        """
        execute an action and return the new pos, observation and battery used
        """
        observation = {'action': action, 'agent_pos': list(agent_pos)}

        if action in self.DIRECTIONS:
            dr, dc = self.DIRECTIONS[action]
            nr, nc = agent_pos[0] + dr, agent_pos[1] + dc

            # bounds and obstacle check
            if (0 <= nr < self.height and 0 <= nc < self.width
                    and self.grid[nr, nc] != 1):
                new_pos = (nr, nc)
                observation['moved'] = True
            else:
                new_pos = agent_pos
                observation['moved'] = False

            observation['new_pos'] = list(new_pos)
            observation['signal'] = self.signal_at(new_pos)

            # check for clues at new position
            if new_pos in self.clues and not self.clues[new_pos]['found']:
                self.clues[new_pos]['found'] = True
                observation['clue'] = {
                    k: v for k, v in self.clues[new_pos].items()
                    if k != 'found'
                }

            # check if eve found
            if new_pos == self.eve_pos:
                observation['found_eve'] = True

            return new_pos, observation, self.battery_per_move

        elif action == 'scan': #eve not found yet so we scan the surrounding area
            observation['signal'] = self.signal_at(agent_pos)
            observation['scan_results'] = []

            for dr in range(-self.scan_radius, self.scan_radius + 1):
                for dc in range(-self.scan_radius, self.scan_radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    sr, sc = agent_pos[0] + dr, agent_pos[1] + dc
                    if 0 <= sr < self.height and 0 <= sc < self.width:
                        cell = {
                            'pos': [sr, sc],
                            'terrain': int(self.grid[sr, sc]),
                        }
                        if ((sr, sc) in self.clues
                                and not self.clues[(sr, sc)]['found']):
                            cell['has_clue'] = True
                        observation['scan_results'].append(cell)

            return agent_pos, observation, self.battery_per_scan

        elif action == 'recharge':
            if self.grid[agent_pos[0], agent_pos[1]] == 2:
                observation['recharged'] = True
                observation['amount'] = self.battery_recharge_amount
            else:
                observation['recharged'] = False
                observation['reason'] = 'not at recharge station'

            observation['signal'] = self.signal_at(agent_pos)
            return agent_pos, observation, 0

        else:
            raise ValueError(f"Unknown action: {action}")

    def get_valid_actions(self, pos, battery):
        #get the list of valid actions that wall-e can do rn
        actions = []

        for action, (dr, dc) in self.DIRECTIONS.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            if (0 <= nr < self.height and 0 <= nc < self.width
                    and self.grid[nr, nc] != 1
                    and battery >= self.battery_per_move):
                actions.append(action)

        if battery >= self.battery_per_scan:
            actions.append('scan')

        if self.grid[pos[0], pos[1]] == 2:
            actions.append('recharge')

        return actions

    def get_initial_signal(self):
        #will return the initial signal that wall-e gets when he starts the mission
        sig = self.config.get('initial_signal', None)
        if sig is None:
            return None
        return {
            'direction': sig.get('direction', 0.0),
            'strength': sig.get('strength', 0.0),
            'timestamp': sig.get('timestamp', 0),
        }