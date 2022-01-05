""" Class definition for particle filter based tempo tracking """
import numpy as np

__author__ = "Matthaios Stylianidis"


class SwitchingKalmanFilterTracker:
    """ Switching Kalman Filter class from tempo tracking - Uses Particle Filtering to estimate switching variable


    Attributes:
        timestamps (list): A list of the timestamps corresponding to the estimated states.
        estimated_states (list): List of estimated states.
        predicted_states (list): List of predicted states.
        score_positions (list): A list of the score positions c_k, used to estimate the score difference.
        score_differences (list): A list of the score difference gamma_k = c_k - c_{k-1}.
        state (np.ndarray): Currently predicted state array containing two values - the next onset time and the
            current tempo estimate. Period is the time between two consecutive beats, not two consecutive notes.
        P (np.ndarraay): State estimate covariance.
        pred_state (np.ndarray): Current predicted state.
        A (np.ndarray): Transition matrix to convert x_{k-1} to x_k, during the prediction stage.
        C (np.ndarray): Matrix for mapping the state variable to an observation.
        K (np.ndarray): The Kalman Gain term.
        switch_variable: The current estimated score difference, used to modify the transition matrix A between
            time steps.
        switch_variable_values (np.ndarray): Numpy array with switch variable discrete possible values
        switch_variable_val_count (int): Number of discrete switch variable possible values
        SWITCH_VARIABLE_RESOLUTION (float): resolution if switch variable discrete values
        SWITCH_VARIABLE_MAX (float): Maximum continuous value of switch variable

    """
    SWITCH_VARIABLE_RESOLUTION = 0.25
    SWITCH_VARIABLE_MAX = 4.0

    def __init__(self,
                 init_tempo_period: float = 1.0,
                 init_state_covariance_std: float = 1.0,
                 onset_process_noise_std: float = 0.001,
                 tempo_process_noise_std: float = 0.001,
                 measurement_noise_std: float = 0.01,
                 particle_no: int = 1000):
        """
        Args:
             start_time (float): Starting time of the track
             init_tempo_period (float): Initial value of tempo period.
             onset_process_noise_std (float): Process standard deviation for onset time.
             tempo_process_noise_std (float): Process standard deviation for tempo.
             measurement_noise_std (float): Measurement model standard deviation.
             particle_no (int): Number of particles for estimating the switching variable.
        """
        self.timestamps = []
        self.estimated_states = []
        self.predicted_states = []
        self.score_positions = []
        self.score_differences = []
        self.switch_variable = 1.0
        self.A = np.array([[1, self.switch_variable], [0, 1]], dtype=np.float64)
        self.C = np.array([1, 0], dtype=np.float64).reshape((1, 2))
        self.state = np.array([0.0, init_tempo_period], dtype=np.float64)
        self.P = np.diag([init_state_covariance_std**2, init_state_covariance_std**2]).astype(dtype=np.float64)
        self.pred_state = None
        self.R = np.diag([onset_process_noise_std**2, tempo_process_noise_std**2]).astype(dtype=np.float64)
        self.Q = np.diag([measurement_noise_std**2]).astype(dtype=np.float64)
        self.K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.Q)
        self.particle_no = particle_no
        self.particles = None
        self.particle_weights = None
        self.switch_variable_state = None
        self.switch_variable_cov = None
        self.switch_variable_val_count = self.SWITCH_VARIABLE_MAX / self.SWITCH_VARIABLE_RESOLUTION
        self.switch_variable_values = np.array([i * self.SWITCH_VARIABLE_RESOLUTION
                                                for i in range(int(self.switch_variable_val_count))], dtype=np.float64)

    def predict(self):
        self.state = self.A @ self.state
        self.pred_state = self.state
        self.P = self.A @ self.P @ self.A.T + self.R
        self.predicted_states.append(self.state)

    def update(self, onset_time):
        """ Estimates state and state uncertainty given a new onset time observation. """
        self.state = self.state + self.K @ (onset_time - self.C @ self.state)
        self.P = self.P - self.K @ self.C @ self.P
        self.estimated_states.append(self.state)

    def compute_kalman_gain(self):
        self.K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.Q)

    def run(self, onset_time):
        # If this is the first iteration, start from update
        if len(self.estimated_states) == 0:
            self.update(onset_time)
        else:
            # Update switch variable
            self.update_switch_variable(onset_time)
            # Predict
            self.predict()
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

        See also the following for MCMC and particle filtering for estimating the switch variable (original paper):
        https://www.aaai.org/Papers/JAIR/Vol18/JAIR-1802.pdf
        """
        # Likelihood of onset time given score difference
        # Multiplied by prior probability of score differences -> Prior here is the prior posititon of the particle
        # filters. Updated each time by the onset time observation to give us the new estimate of the score difference

        # Get ideal score difference based on onset difference and previously estimated tempo
        prev_estimated_onset, prev_estimated_tempo = self.estimated_states[-1]
        ideal_score = (onset_time - prev_estimated_onset) / prev_estimated_tempo

        # Round ideal score to closest discrete switch variable value
        rounded_ideal_score = self.switch_variable_values [np.abs(self.switch_variable_values - ideal_score).argmin()]

        self.switch_variable = rounded_ideal_score
        self.A[0, 1] = self.switch_variable


        # Search in local neighborhood of switch variable for

        # Rao blackwellized particle filter
        # Estimate switch variable gamma_k integrating over the possible tempos z_k
        # For each possible tempo, sample switch variable gamma_K


        # The swithc variable in a time step can have S distinct states and we wish to generate N samples.

        # The switch variable here can be naively estimated by running the Kalman Filter S times on the observation
        # sequence to calculate the proposal of gamma_k in the current step given all the previous proposals and onsets
        # integrating out the  tempi

        # Cemgil for each trajectory the integration of the tempo is computed stepwise by the Kalman Filter
        # However to find the MAP estimate of equation 11 we need to evaluate the probability of the onsets
        # given the series of score differences until now, independently for all exponentially many trajectories.
        # Consequently the MAP estimation for this function can only be solved approximately

        # This is a combinatorial optimization problem: we seek the maximum of a function p(γ1:K|y0:K)
        # that associates a number with each of the discrete configurations γ1:K.

        # The first important observation is that, conditioned on γ1:K, the model becomes a linear
        # state space model and the integration on z0:K can be computed analytically using Kalman
        # filtering equations

        # According to Cemgil, it seems like that particles are initialized to (tempo, score_diff) space, and
        # the different tempos are integrated out by greedily expanding from N to N x S states and then keeping
        # only N states
        #
        #
        pass
        # Rao-Blackwellized (integrating tempos out) particle filter for Switching State space Model

        pass

    def greedy_expansion(self):
        """ Expansion of the particle set according to Greedy Filtering as described in
        https://www.aaai.org/Papers/JAIR/Vol18/JAIR-1802.pdf

        In each timestep, this function is called  to expand the set of N particles to obtain N x S new particles,
        where S is the number of possible score differences.
        """
        pass

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

    @staticmethod
    def tempo_period_to_bpm(period):
        return 60.0 / period
