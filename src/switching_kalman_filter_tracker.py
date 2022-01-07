""" Class definition for particle filter based tempo tracking """
import numpy as np

__author__ = "Matthaios Stylianidis"


class SwitchingKalmanFilterTracker:
    """ Switching Kalman Filter class from tempo tracking - Uses Particle Filtering to estimate switching variable


    Attributes:
        estimated_states (list): List of estimated states. If a PF is used then the particle with the maximum likeli-
            hood is chosen.
        estimated_covs (list): List of estimated covariance. If a PF is used then the particle with the maximum
            likelihood is chosen.
        score_differences (list): A list of the score difference gamma_k = c_k - c_{k-1}.
        state (np.ndarray): Currently predicted state array containing two values - the next onset time and the
            current tempo estimate. Period is the time between two consecutive beats, not two consecutive notes.
        P (np.ndarraay): State estimate covariance.
        A (np.ndarray): Transition matrix to convert x_{k-1} to x_k, during the prediction stage.
        C (np.ndarray): Matrix for mapping the state variable to an observation.
        K (np.ndarray): The Kalman Gain term.
        switch_variable_values (np.ndarray): Numpy array with switch variable discrete possible values
        switch_variable_val_count (int): Number of discrete switch variable possible values
        SWITCH_VARIABLE_RESOLUTION (float): resolution if switch variable discrete values
        SWITCH_VARIABLE_MAX (float): Maximum continuous value of switch variable
        use_ideal_switch_value (bool): If set to true the ideal switch variable value is used and particle filter-
                ing is not employed.
        particle_no (int): Number of non-expanded particles if particle filtering is used.
        expand_factor (int): Number of new states predicted given each particle. In each iteration the particles
            are expanded to a set of expand_factor X particle_no new particles, and are then pruned.
        expanded_state  (np.ndarray): Numpy array that contains all the predicted states after expanding each particle
            to <expand_factor> new states using prediction.
        expanded_P  (np.ndarray): Numpy array that contains all the expanded state uncertainties.
    """
    SWITCH_VARIABLE_RESOLUTION = 0.25
    SWITCH_VARIABLE_MAX = 4.0

    def __init__(self,
                 init_tempo_period: float = 1.0,
                 init_state_covariance_std: float = 1.0,
                 onset_process_noise_std: float = 0.001,
                 tempo_process_noise_std: float = 0.001,
                 measurement_noise_std: float = 0.01,
                 particle_no: int = 10,
                 expand_factor: int = 5,
                 use_ideal_switch_value: bool = False):
        """
        Args:
             start_time (float): Starting time of the track
             init_tempo_period (float): Initial value of tempo period.
             onset_process_noise_std (float): Process standard deviation for onset time.
             tempo_process_noise_std (float): Process standard deviation for tempo.
             measurement_noise_std (float): Measurement model standard deviation.
             particle_no (int): Number of particles for estimating the switching variable.
        """
        self.estimated_states = []
        self.estimated_covs = []
        self.score_differences = []
        self.A = np.array([[1, 1.0], [0, 1]], dtype=np.float64)
        self.C = np.array([1, 0], dtype=np.float64).reshape((1, 2))
        self.state = np.array([0.0, init_tempo_period], dtype=np.float64)
        self.P = np.diag([init_state_covariance_std**2, init_state_covariance_std**2]).astype(dtype=np.float64)
        self.R = np.diag([onset_process_noise_std**2, tempo_process_noise_std**2]).astype(dtype=np.float64)
        self.Q = np.diag([measurement_noise_std**2]).astype(dtype=np.float64)
        self.K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.Q)
        self.particle_no = particle_no
        self.particles = None
        self.particle_weights = None
        self.switch_variable_state = None
        self.switch_variable_cov = None
        self.switch_variable_val_count = int(self.SWITCH_VARIABLE_MAX / self.SWITCH_VARIABLE_RESOLUTION)
        self.switch_variable_values = np.array([i * self.SWITCH_VARIABLE_RESOLUTION
                                                for i in range(self.switch_variable_val_count)], dtype=np.float64)
        self.use_ideal_switch_value = use_ideal_switch_value
        self.expand_factor = expand_factor

        # Initialize particle set
        if not self.use_ideal_switch_value:
            self.particle_no = particle_no
            self.particles = self.init_particles(self.particle_no)
            self.particle_weights = np.full((self.particle_no, 1), 1/self.particle_no)
            # Create as many replicas of the Kalman Filter variables as the number of particles
            self.state = np.tile(self.state, (self.particle_no, 1))[..., np.newaxis]
            self.A = np.tile(self.A, (self.particle_no, 1, 1))
            self.A[:, 1, 1] = self.particles
            self.P = np.tile(self.P, (self.particle_no, 1, 1))
            self.K = np.tile(self.K.T, (self.particle_no, 1))[..., np.newaxis]
            self.expanded_state = np.tile(self.state, (self.expand_factor, 1, 1))
            self.expanded_P = np.tile(self.P, (self.expand_factor, 1, 1))

    def init_particles(self, particle_no):
        """ Initializes particle distribution """
        # Sample uniformly with replacement from the grid of possible score differences
        return np.random.choice(self.switch_variable_values, particle_no)

    def predict(self):
        if self.use_ideal_switch_value:
            self.state = self.A @ self.state
            self.P = self.A @ self.P @ self.A.T + self.R
        else:
            # For each particle
            for i in range(self.particle_no):
                # Take expand_factor samples around ideal rounded score difference, which is stored in A
                rounded_score_difference = self.A[i, 0, 1]
                score_diff_index = np.argwhere(self.switch_variable_values == rounded_score_difference)[0, 0]
                score_indices = [score_diff_index - self.expand_factor // 2 + i for i in range(self.expand_factor)]
                # If index went out of the grid's indices, replace it with the ideal score index
                score_indices = map(lambda x: min(0, max(self.switch_variable_val_count, x)), score_indices)

                # For each score index computed, expand state
                for j, score_index in enumerate(score_indices):
                    switch_variable = self.switch_variable_values[score_index]
                    expanded_index = self.expand_factor * i + j
                    self.A[i, 0, 1] = switch_variable
                    self.expanded_state[expanded_index] = self.A[i] @ self.state[i]
                    self.expanded_P[expanded_index] = self.A[i] @ self.P[i] @ self.A[i].T + self.R

    def prune_states(self, onset_time):
        # Calculate likelihoods
        likelihoods = np.zeros(self.expanded_state.shape[0])
        for i, expanded_state in enumerate(self.expanded_state):
            residual = onset_time - self.C @ expanded_state
            covE = (self.C @ self.expanded_P[i] @ self.C.T + self.R)[0, 0]
            likelihoods[i] = (1 / np.sqrt(covE * 2 * np.pi)) * np.exp(-0.5 * residual ** 2 * covE ** -1)
        # Get the particles with the maximum likelihood
        max_likelihood_arg = np.argpartition(likelihoods, -self.particle_no)[-self.particle_no:]
        for i in range(self.particle_no):
            self.state[i] = self.expanded_state[max_likelihood_arg[i]]
            self.P[i] = self.expanded_P[max_likelihood_arg[i]]

    def update(self, onset_time):
        """ Estimates state and state uncertainty given a new onset time observation. """
        if self.use_ideal_switch_value:
            self.state = self.state + self.K @ (onset_time - self.C @ self.state)
            self.P = self.P - self.K @ self.C @ self.P
            self.estimated_states.append(self.state)
        else:
            for i in range(self.particle_no):
                self.state[i] = self.state[i] + self.K[i] @ (onset_time - self.C @ self.state[i])
                self.P[i] = self.P[i] - self.K[i] @ self.C @ self.P[i]
            # Assign particle with maximum probability to estimated states
            max_prob_particle = np.argmax(self.particle_weights)
            self.estimated_states.append(self.state[max_prob_particle])
            self.estimated_covs.append(self.P[max_prob_particle])

    def compute_kalman_gain(self):
        if self.use_ideal_switch_value:
            self.K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.Q)
        else:
            for i in range(self.particle_no):
                self.K[i] = self.P[i] @ self.C.T @ np.linalg.inv(self.C @ self.P[i] @ self.C.T + self.Q)

    def run(self, onset_time):
        # If this is the first iteration, start from update
        if len(self.estimated_states) == 0:
            self.update(onset_time)
        else:
            # Update switch variable
            self.update_switch_variable(onset_time)
            # Predict
            self.predict()

            # Prune expanded states according to likelihood if PF is used
            if not self.use_ideal_switch_value:
                self.prune_states(onset_time)

            # Update Kalman Gain
            self.compute_kalman_gain()
            # Update
            self.update(onset_time)

    def get_estimates(self):
        return self.estimated_states

    def get_tempo_estimates(self):
        return np.array([x[1] for x in self.estimated_states], dtype=np.float64)

    def update_switch_variable(self, onset_time):
        """ Update switch value according to particle filtering

        According to the derivation in https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.217.32&rep=rep1&type=pdf
        the ideal switch variable value for which the residual e_k is zero can be calculated as follows:

        e_k (residual) = y_k - C x_pred_{k} = y_k - C A x_{k-1} =  0 => (y_k - r_{k-1}) / Delta_{k-1} = gamma_k

        where Delta is the tempo period, gamma is the score difference, and y_k is the observation (onset time).

        However, we use particle filtering to estimate and track the switch variable since it is a random discrete vari
        able.
        """
        if self.use_ideal_switch_value:
            prev_estimated_onset, prev_estimated_tempo = self.state
            ideal_score = (onset_time - prev_estimated_onset) / prev_estimated_tempo
            rounded_ideal_score = self.switch_variable_values[
                np.abs(ideal_score - self.switch_variable_values).argmin()]
            self.A[0, 1] = rounded_ideal_score
        else:
            # For each one of the last particle states, calculate rounded ideal score and store in A
            for i in range(self.particle_no):
                prev_estimated_onset, prev_estimated_tempo = self.state[i]
                ideal_score = (onset_time - prev_estimated_onset) / prev_estimated_tempo
                rounded_ideal_score = self.switch_variable_values[
                    np.abs(ideal_score - self.switch_variable_values).argmin()]
                self.A[i, 0, 1] = rounded_ideal_score

    @staticmethod
    def tempo_period_to_bpm(period):
        return 60.0 / period

    def systematic_resampling(self):
        """ Systematic resampling for particles """
        # Calculate CDF of weights
        new_particle_set = np.zeros_like(self.particles)
        CDF = np.cumsum(self.particle_weights)
        r_0 = np.random.random()
        for m in range(self.particle_no):
            r = min(r_0 + (m - 1) / self.particle_no, 1.0)
            i = np.searchsorted(CDF, r)
            new_particle_set[m] = self.particles[i]
        self.particles = new_particle_set
        self.particle_weights = 1 / self.particle_no

