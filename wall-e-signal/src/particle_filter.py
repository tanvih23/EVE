#the belief tracker, basically a bunch of guesses (particles) about where eve is
#that get reweighted every time wall-e gets a signal reading.

import numpy as np
from collections import defaultdict


class ParticleFilter:
    def __init__(self, n_particles, valid_cells, signal_noise_std,
                 signal_base=None):
        self.n_particles = n_particles
        self.signal_noise = signal_noise_std
        self.signal_base = signal_base
        self.valid_cells = valid_cells
        self.valid_set = set(valid_cells)

        # grid bounds, so we can clip roughened particles back in-world
        vc = np.array(valid_cells)
        self.max_row = int(vc[:, 0].max())
        self.max_col = int(vc[:, 1].max())

        # tiny jitter added after resampling so the filter never freezes onto
        # kept small so a converged belief still reads as high confidence
        self.roughen_floor = 0.08

        
        self.under_tolerance = 1.3
        self.robust_clip = 4.0

        # scatter particles uniformly
        indices = np.random.choice(len(valid_cells), size=n_particles)
        self.particles = np.array(
            [valid_cells[i] for i in indices], dtype=float
        )
        self.weights = np.ones(n_particles) / n_particles

        # keep belief history for analysis
        self.history = []

    def update_signal(self, observed_signal, agent_pos, signal_model_fn):
        #bayes update: new weight = old weight x likelihood of the reading
        #under each particle's guess then normalize & particles that match
        #the reading survive ones that dont get killed of
        ar, ac = agent_pos

        # distance^2 from agent to each particle
        dist_sq = (self.particles[:, 0] - ar)**2 + (self.particles[:, 1] - ac)**2

        # expected signal under each hypothesis 
        if self.signal_base is not None:
            # fast path one vectorized expression no python loop
            expected = self.signal_base / (1.0 + dist_sq)
        else:
            # fallback for a custom forward model
            expected = np.zeros(self.n_particles)
            for i in range(self.n_particles):
                expected[i] = signal_model_fn(
                    agent_pos, tuple(self.particles[i].astype(int))
                )

    
        z = (observed_signal - expected) / self.signal_noise
        scale = np.where(z >= 0, self.under_tolerance, 1.0)
        z_clipped = np.clip(z / scale, -self.robust_clip, self.robust_clip)
        likelihoods = np.exp(-0.5 * z_clipped**2)

        # update weights
        self.weights *= likelihoods

        # normalize
        total = self.weights.sum()
        if total > 1e-300:
            self.weights /= total
        else:
            # everything collapsed if all guesses equally bad then reset to uniform
            self.weights = np.ones(self.n_particles) / self.n_particles

        # resample if too few particles are actually contributing
        n_eff = self.effective_sample_size()
        if n_eff < self.n_particles * 0.5:
            self.systematic_resample()

        # record state
        self.history.append({
            'estimate': self.estimate().tolist(),
            'confidence': float(self.confidence()),
            'n_eff': float(n_eff),
            'entropy': float(self.grid_entropy()),
        })

    def update_clue(self, clue, clue_pos):
        #a clue points roughly toward eve so upweight particles that sit in
        #that direction from the clue uses a von mises style likelihood
        #L = e^(k x cos(angle)) where k is how tight the cone is
        if clue.get('direction_hint') is None:
            return

        direction_rad = np.radians(clue['direction_hint'])
        # unit vector in the hint direction
        target_dir = np.array([
            np.sin(direction_rad),
            np.cos(direction_rad),
        ])

        clue_arr = np.array(clue_pos, dtype=float)

        for i in range(self.n_particles):
            to_particle = self.particles[i] - clue_arr
            dist = np.linalg.norm(to_particle)
            if dist < 0.5:
                continue
            to_particle /= dist

            # cosine similarity 1 if directions match and -1 if opposite
            cos_angle = np.dot(to_particle, target_dir)
            # k=2 is a moderately peaked cone
            self.weights[i] *= np.exp(2.0 * cos_angle)

        # normalize
        total = self.weights.sum()
        if total > 1e-300:
            self.weights /= total

        # resample if needed
        if self.effective_sample_size() < self.n_particles * 0.5:
            self.systematic_resample()

    def update_initial_signal(self, initial_signal, agent_pos):
        #the one coarse bearing wall-e starts the mission with same idea as
        #a clue but weaker (k=1) since its noisy
        if initial_signal is None:
            return

        direction_rad = np.radians(initial_signal['direction'])
        target_dir = np.array([
            np.sin(direction_rad),
            np.cos(direction_rad),
        ])
        agent_arr = np.array(agent_pos, dtype=float)

        for i in range(self.n_particles):
            to_particle = self.particles[i] - agent_arr
            dist = np.linalg.norm(to_particle)
            if dist < 0.5:
                continue
            to_particle /= dist
            cos_angle = np.dot(to_particle, target_dir)
            # weaker constraint since the initial reading is noisy
            self.weights[i] *= np.exp(1.0 * cos_angle)

        total = self.weights.sum()
        if total > 1e-300:
            self.weights /= total

        if self.effective_sample_size() < self.n_particles * 0.5:
            self.systematic_resample()

    # core particle filter ops

    def effective_sample_size(self):
        #N_eff = 1 / sum(w^2). rough count of how many particles actually
        #matter close to N = healthy close to 1 = one particle hogging all
        #the weight (degeneracy) if below N/2 then resample
        return 1.0 / np.sum(self.weights**2)

    def systematic_resample(self):
        #replace low-weight particles with copies of high-weight ones using
        #one random offset and N evenly spaced picks along the weight CDF
        #more even than drawing N independent samples and its O(N) — the
        #standard trick in robotics.
        N = self.n_particles
        cumsum = np.cumsum(self.weights)

        # one random starting point
        u = np.random.random() / N
        # N evenly spaced points
        positions = u + np.arange(N) / N

        # walk the CDF to see which particle each point lands on
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, N - 1)

        # replace particles
        self.particles = self.particles[indices].copy()
        self.weights = np.ones(N) / N

        self._roughen()

    def _roughen(self):
        #add a bit of gaussian noise to the particles after resampling
        std = self.particles.std(axis=0)
        bw = 0.5 * std * self.n_particles ** (-1.0 / 6.0)
        bw = np.maximum(bw, self.roughen_floor)
        self.particles += np.random.normal(0.0, 1.0, self.particles.shape) * bw
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.max_row)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.max_col)

    # belief queries

    def estimate(self):
        #weighted mean of the particles = best guess for eve's position
        return np.average(self.particles, weights=self.weights, axis=0)

    def confidence(self):
        #how tight the particles are. confidence = 1/(1+sqrt(variance)) in interval
        #(0,1] near 1 = very sure, near 0 = clueless
        mean = self.estimate()
        variance = np.average(
            np.sum((self.particles - mean)**2, axis=1),
            weights=self.weights,
        )
        return 1.0 / (1.0 + np.sqrt(variance))

    def grid_entropy(self):
        #entropy of the belief if you bin the particles into a grid useful
        #for measuring how much the belief has collapsed after a signal reading
        r = np.clip(self.particles[:, 0], 0, 1e6).astype(np.int64)
        c = np.clip(self.particles[:, 1], 0, 1e6).astype(np.int64)
        keys = r * 100003 + c  # 100003 > any realistic column count
        _, inv = np.unique(keys, return_inverse=True)
        cell_w = np.bincount(inv, weights=self.weights)
        p = cell_w[cell_w > 1e-10]
        return float(-np.sum(p * np.log2(p)))

    def top_cells(self, k=5):
        #the k most likely cells. handy for spotting when the
        #belief is split between two spots.
        cell_weights = defaultdict(float)
        for i in range(self.n_particles):
            r = int(self.particles[i, 0])
            c = int(self.particles[i, 1])
            cell_weights[(r, c)] += self.weights[i]

        sorted_cells = sorted(
            cell_weights.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_cells[:k]

    def hypothetical_update(self, observed_signal, agent_pos, signal_model_fn):
        #return a copy of the filter after a hypothetical update with the
        #given signal reading useful for evaluating the expected information
        clone = ParticleFilter.__new__(ParticleFilter)
        clone.n_particles = self.n_particles
        clone.particles = self.particles.copy()
        clone.weights = self.weights.copy()
        clone.signal_noise = self.signal_noise
        clone.signal_base = self.signal_base
        clone.valid_cells = self.valid_cells
        clone.valid_set = self.valid_set
        clone.max_row = self.max_row
        clone.max_col = self.max_col
        clone.roughen_floor = self.roughen_floor
        clone.under_tolerance = self.under_tolerance
        clone.robust_clip = self.robust_clip
        clone.history = []

        clone.update_signal(observed_signal, agent_pos, signal_model_fn)
        return clone

    def density_map(self, height, width):
        #bin the particles into a 2d probability map for the heatmap
        density = np.zeros((height, width))
        for i in range(self.n_particles):
            r = int(np.clip(self.particles[i, 0], 0, height - 1))
            c = int(np.clip(self.particles[i, 1], 0, width - 1))
            density[r, c] += self.weights[i]
        return density