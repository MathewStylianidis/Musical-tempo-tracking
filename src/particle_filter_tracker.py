""" Particle filter solution for tempo tracking using a stream of discrete onset times. This tracker models the score
 difference as a hidden variable. """
import numpy as np


class ParticleFilterTracker:
    """ Particle filter tempo tracker and rhythm quantizer.

    Attributes:
        base_particle_no (int): The number of particles before expanding each particle to S particles with the same
            tempo and all the possible score difference values.
        total_particle_no (int): The number of particles after expanding each particle to S particles with the same
            tempo and all the possible difference values.
        particles (np.ndarray): Numpy array of particles where each particle tracks a state with an onset value, a tempo
            period and score difference.
        min_tempo_period (float): Minimum tempo period that can be tracked.
        max_tempo_period (float): Maximum tempo period that can be tracked.
        switch_variable_values (np.ndarray): Numpy array with switch variable discrete possible values
        switch_variable_val_count (int): Number of discrete switch variable possible values
        SWITCH_VARIABLE_RESOLUTION (float): resolution if switch variable discrete values
        SWITCH_VARIABLE_MAX (float): Maximum continuous value of switch variable
        estimated_states (list): List of all the estimated states with the maximum likelihood from each iteration.
        A (np.ndarray): Transition matrix to convert x_{k-1} to x_k, during the prediction stage.
        R (np.ndarray): Process noise covariance matrix with 2x2 dimensions (1 dimension for the onset and one for
            the tempo period).
        Q (np.ndarray): Measurement/Observation (onset time) noise variance.
        likelihoods (np.ndarray): Conditional probability of <onset, tempo, score difference> pair given the onset.
        marginal_likelihoods (np.ndarray): Marginal probability of <onset, tempo> pair after maginalizing over the score
            difference values.
    """
    SWITCH_VARIABLE_RESOLUTION = 0.25
    SWITCH_VARIABLE_MAX = 4.0

    def __init__(self, base_particle_no: int = 100,
                 min_tempo: int = 50,
                 max_tempo: int = 200,
                 onset_process_noise_std: float = 0.001,
                 tempo_process_noise_std: float = 0.001,
                 measurement_noise_std: float = 0.01):
        # Initialize particles
        self.base_particle_no = base_particle_no
        self.min_tempo_period = max_tempo / 60.0
        self.max_tempo_period = min_tempo / 60.0
        self.switch_variable_val_count = int(self.SWITCH_VARIABLE_MAX / self.SWITCH_VARIABLE_RESOLUTION) + 1
        self.switch_variable_values = np.array([i * self.SWITCH_VARIABLE_RESOLUTION
                                                for i in range(self.switch_variable_val_count)], dtype=np.float64)

        self.particles = self.init_particles()
        self.total_particle_no = len(self.particles)

        self.estimated_states = []
        self.A = np.array([[1, 1], [0, 1]], dtype=np.float64)
        self.R = np.diag([onset_process_noise_std**2, tempo_process_noise_std**2]).astype(dtype=np.float64)
        self.Q = np.diag([measurement_noise_std ** 2]).astype(dtype=np.float64)
        self.likelihoods = np.full(self.total_particle_no, 1 / self.total_particle_no)
        self.marginal_likelihoods = np.full(self.base_particle_no, 1 / self.base_particle_no)

        # Precalculate inverse and determinant of Q
        self._inv_Q = self.Q ** (-1)
        self._det_Q = np.linalg.det(self.Q)
        self._likelihood_norm_const = 1 / (2 * np.pi * np.sqrt(self._det_Q))

    def init_particles(self):
        """ Initializes particle set.

        Returns:
            Numpy array with (<base_particle_no> x <switch_variable_val_count>) x 3 dimensions representing the state
                tracked by each particle. Each state is initialized with a 0.0 onset, a tempo drawn from a uniform
                random distribution and a discrete score difference value drawn from the grid. The returned particle
                distribution samples base_particle_no tempos and then makes sure the full score difference distribution
                is represented by expanding each of those particles to <switch_variable_val_count> particles with the
                same tempo value and all score difference values.

        """
        # Initialize onsets to 0.0 and sample tempo from uniform distribution
        onset_times = np.full(self.base_particle_no, 0.0)
        score_differences = np.full(self.base_particle_no, 0.0)
        tempos = np.random.uniform(low=self.min_tempo_period, high=self.max_tempo_period, size=self.base_particle_no)
        onset_tempo_pairs = np.vstack([onset_times, tempos, score_differences]).T
        # Expand each particle to <switch_variable_val_count> new particles with the same tempo but all possible
        # score differences.
        particles = np.tile(onset_tempo_pairs, reps=(self.switch_variable_val_count, 1))
        for i in range(self.switch_variable_val_count):
            particles[self.base_particle_no * i:self.base_particle_no * i + self.switch_variable_val_count, 2] \
                = self.switch_variable_values[i]
        return particles

    def run(self, onset_time: float):
        """ Runs the particle filter for one iteration"""
        # Predict next onset based on score difference/tempo of each particle
        self.predict()
        # Compute weights, prune and expand particle set - Update
        self.compute_weights(onset_time)
        self.prune_and_expand()

    def compute_weights(self, onset_time: float):
        """ Computes likelihood given the observation of each <onset, tempo, score_difference> pair"""
        for i in range(self.total_particle_no):
            exponent = -0.5 * (onset_time - self.particles[i, 0]) ** 2 * self._inv_Q
            self.likelihoods[i] = self._likelihood_norm_const * np.exp(exponent)
        #for i in range(self.switch_variable_val_count):
            #for j in range(self.base_particle_no):
                #self.marginal_likelihoods[j] += self.likelihoods[i * self.base_particle_no + j]

    def prune_and_expand(self):
        """ Selects <base_particle_no> particles with the highest marginal likelihoods given the onset, and expands
        those particles to <total_particle_no> new particles where again all possible score difference values are
        covered by the new particle set. """
        # Prune all but the most likely particles
        best_particle_indices = np.argpartition(self.likelihoods,
                                                -self.base_particle_no)[-self.base_particle_no:]
        self.particles = self.particles[best_particle_indices]
        # Store highest likelihood particle as estimate
        # arg_max_marginal_likelihood = np.argmax(self.marginal_likelihoods)
        # max_likelihood_particle = self.particles[arg_max_marginal_likelihood]
        highest_likelihoods = self.likelihoods[best_particle_indices]
        max_likelihood_particle = self.particles[np.argmax(highest_likelihoods)]
        self.estimated_states.append(max_likelihood_particle)
        # Expand
        self.particles = np.tile(self.particles, reps=(self.switch_variable_val_count, 1))
        for i in range(self.switch_variable_val_count):
            self.particles[self.base_particle_no * i:self.base_particle_no * i + self.switch_variable_val_count, 2] \
                = self.switch_variable_values[i]

    def update_transition_matrix(self, score_difference: float):
        """ Updates the transition matrix given the score difference

        Attribute:
            score_difference (float): The switch variable according to which the transition matrix should be
                updated.
        """
        self.A[0, 1] = score_difference

    def predict(self):
        """ Predict function of the particle filter. """
        for i in range(len(self.particles)):
            self.update_transition_matrix(self.particles[i, 2])
            noise = np.random.multivariate_normal([0.0 for _ in range(len(self.R))], self.R, 1)
            self.particles[i, :2] = self.A @ self.particles[i, :2] + noise

    def get_estimates(self):
        return self.estimated_states

    def get_tempo_estimates(self):
        return np.array([x[1] for x in self.estimated_states], dtype=np.float64)

    @staticmethod
    def tempo_period_to_bpm(period):
        return 60.0 / period


