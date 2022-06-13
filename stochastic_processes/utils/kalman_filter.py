import numpy as np
import numba as nb
import pandas as pd

def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            print('linalgerror', A)
            return False
    else:
        print('not equal')
        return False

def change_matrix_to_pos_def(M):
    M[M<0] = 0
    return M

def _cast_float2mat(element):
    if isinstance(element, float): 
        return np.array([[element]], dtype = np.float64)
    elif isinstance(element, int): 
        return np.array([[float(element)]], dtype = np.float64)
    elif isinstance(element, np.ndarray):
        if (len(element.shape) == 2):
            return element
        elif (len(element.shape) == 1):
            return element.reshape(1, -1)
    elif isinstance(element, pd.Series):
        return element.values.reshape(1, -1)
    raise TypeError

class KalmanFilter(object):
    def __init__(self,
        curr_state,
        curr_measurement_noise_cov,
        curr_state_estimate_cov,
        curr_state_noise_cov,
        state_to_state_transition_mat = None, *args, **kwargs):
        """
        Example:
        1d case
        >>> initial_state = np.array([[0]])
        >>> initial_measurement_noise_cov = 0.1**2
        >>> initial_state_estimate_cov = 1 * np.eye(1)
        >>> initial_state_noise_cov = 1e-5 * np.eye(1)
        >>> kf = AdaptiveExtendedKalmanFilter(curr_measurement_noise_cov=1e-5, curr_state = np.array([[1e-3]]), curr_state_estimate_cov=1e-7 * np.eye(1), curr_trans_cov=1e-9 * np.eye(1), measurement_decay=.8, state_decay=.8)
        2d case
        TODO
        """
        self.curr_state = _cast_float2mat(curr_state)
        self.curr_state_estimate_cov = _cast_float2mat(curr_state_estimate_cov)
        self.curr_measurement_noise_cov = _cast_float2mat(curr_measurement_noise_cov)
        self.curr_state_noise_cov = _cast_float2mat(curr_state_noise_cov)
        n,_ = self.curr_state_estimate_cov.shape
        if state_to_state_transition_mat is None:
            self.state_to_state_transition_mat = np.identity(n)
        else:
            self.state_to_state_transition_mat = state_to_state_transition_mat
    
    def predict(self, X: np.ndarray):
        return  _cast_float2mat(X) @ self.curr_state
    
    def fit_once(self, X: np.array, y: float):
        self.curr_state, self.curr_state_estimate_cov, *info= self._each_step(
            X = _cast_float2mat(X),
            y = _cast_float2mat(y), 
            prev_state = self.curr_state,
            prev_measurement_noise_cov = self.curr_measurement_noise_cov,
            prev_state_estimate_cov = self.curr_state_estimate_cov,
            prev_state_noise_cov = self.curr_state_noise_cov,
            state_to_state_transition_mat = self.state_to_state_transition_mat
        )
        return info

    def fit_all(self, Xarr: np.array, yarr: np.array, info = None):
        """
        X: (n_iter, n, n) where n is the number of states
        y: (n_iter, ndim, 1) where ndim is the number of dimension 
        """
        series = []
        for i in range(yarr.shape[0]):
            info = self.fit_once(Xarr[i], yarr[i])
            series.append(info)
        return series
    
    # def fit_all(self, X: pd.DataFrame, y: pd.Series, info = None):
    #     """
    #     X: (n_iter, n, n) where n is the number of states
    #     y: (n_iter, ndim) where ndim is the number of dimension 
    #     """
    #     series = []
    #     index = X.index
    #     Xarr = X.values
    #     yarr = y.values
    #     for i in range(y.shape[0]):
    #         info = self.fit_once(Xarr[i], yarr[i])
    #         series.append(info)
    #     series = pd.DataFrame(series, index = index)
    #     return series
    
    @staticmethod
    def _each_step(X, y,
        prev_state,
        prev_measurement_noise_cov,
        prev_state_estimate_cov,
        prev_state_noise_cov,
        state_to_state_transition_mat):
        predicted_state = state_to_state_transition_mat @ prev_state
        predicted_state_estimate_cov = state_to_state_transition_mat @ prev_state_estimate_cov @ state_to_state_transition_mat.T + prev_state_noise_cov
    
        kalman_gain = predicted_state_estimate_cov @ X.T @ np.linalg.inv(
            (X @ predicted_state_estimate_cov @ X.T + prev_measurement_noise_cov)
            )
        next_state = predicted_state + kalman_gain @ (y - (X @ predicted_state))
        n, _ = kalman_gain.shape
        next_state_estimate_cov = (np.eye(n) - kalman_gain @ X) @ predicted_state_estimate_cov
        info = [next_state, next_state_estimate_cov]
        return next_state, next_state_estimate_cov, info
    
class AdaptiveExtendedKalmanFilter(KalmanFilter):
    """
    Adaptive Extended Kalman filter using decay factor for Q and R
        https://arxiv.org/ftp/arxiv/papers/1702/1702.00884.pdf
    with abnormal noise detection in INS/GNSS
        https://www.sciencedirect.com/science/article/pii/S1000936121001667
    Phi is automatically set to be identidy matrix because the state transition is identical
    """
    def __init__(
        self,
        curr_measurement_noise_cov,
        curr_state,
        curr_state_estimate_cov,
        curr_state_noise_cov,
        measurement_decay = 0.3,
        state_decay = 0.3,
        gamma = 15.,
        *args,
        **kwargs
    ):
        """
        Example:
        >>> y_dim = 1
        >>> num_features = 2
        >>> akf = AdaptiveExtendedKalmanFilter(
            curr_measurement_noise_cov = 1e-5 * np.eye(y_dim), 
            curr_state = 1e-3 * np.ones(num_features),
            curr_state_estimate_cov = 1e-7 * np.eye(num_features),
            curr_state_noise_cov = 1e-9 * np.eye(num_features),
            measurement_decay=.8, state_decay=.8)
       """
        self.curr_state = _cast_float2mat(curr_state)
        self.curr_state_estimate_cov = _cast_float2mat(curr_state_estimate_cov)
        self.curr_measurement_noise_cov = _cast_float2mat(curr_measurement_noise_cov)
        self.curr_state_noise_cov = _cast_float2mat(curr_state_noise_cov)
        self.measurement_decay = measurement_decay
        self.state_decay = state_decay
        self.curr_prediction = np.nan
        self.gamma = gamma

    def fit_once(self, X: np.array, y: float):
        self.curr_prediction, self.curr_state, self.curr_state_estimate_cov, self.curr_measurement_noise_cov, self.curr_state_noise_cov, *info= self._each_step(
            _cast_float2mat(X),
            _cast_float2mat(y),
            last_beta = self.curr_state,
            last_posterior_cov = self.curr_state_estimate_cov,
            last_obs_cov = self.curr_measurement_noise_cov,
            last_trans_cov = self.curr_state_noise_cov,
            measurement_decay = self.measurement_decay,
            state_decay = self.measurement_decay,
            gamma = self.gamma
        )
        return info

    @staticmethod
    def _each_step(
        X,
        y,
        last_beta,
        last_posterior_cov,
        last_obs_cov,
        last_trans_cov,
        measurement_decay,
        state_decay,
        gamma,
    ):
        # step 1
        pred_beta = last_beta # prior beta
        predict_y = X @ pred_beta # prior y
        prior_cov = last_posterior_cov + last_trans_cov # prior P_k
        # step 2
        measurement_inno = y - predict_y # d_t
        inno_cov = X @ prior_cov @ X.T + last_obs_cov # S_k
        kalman_gain = (prior_cov @ X.T @ np.linalg.inv(inno_cov)) * np.heaviside(gamma * np.trace(inno_cov) - np.trace(measurement_inno * measurement_inno.T), 1.) # K_k * abnormal measurement detection
        posterior_beta = pred_beta + kalman_gain @ measurement_inno # posterior beta
        posterior_cov = prior_cov - kalman_gain @ X @ prior_cov # posterior P_k
        residual = y - X @ posterior_beta # epsilon
        obs_cov = (
                measurement_decay * last_obs_cov
                + (1 - measurement_decay) * (residual @ residual.T + X @ prior_cov @ X.T) # R_k update
            )
        trans_cov = (
            state_decay * last_trans_cov +
            (1 - state_decay) * (kalman_gain @ measurement_inno  @ measurement_inno.T @ kalman_gain.T) # Q_k update
            )

        # temporarily store information
        info = [posterior_beta, posterior_cov, obs_cov, trans_cov]
        return predict_y, posterior_beta, posterior_cov, obs_cov, trans_cov, info # take abs because the determinant may not be positive definite

class KLDMinimizationBasedAKF(KalmanFilter):
    def __init__(self,
        curr_state,
        curr_measurement_noise_cov,
        curr_state_estimate_cov,
        curr_state_noise_cov,
        dof=5,
        state_to_state_transition_mat = None,
        gamma = 1.345,
        N_m = 100,
        epsilon = 1e-99,
        N_m_prime = 50,
        epsilon_prime = 1e-17,
        tau_k = 0.15,
        lambda_k = 0.15,
        *args,
        **kwargs
    ):
        """
        Example:
        1d case
        >>> akf = AdaptiveExtendedKalmanFilter(curr_measurement_noise_cov=1e-5, curr_state = np.array([[1e-3]]), curr_state_estimate_cov=1e-7 * np.eye(1), curr_state_noise_cov=1e-9 * np.eye(1), measurement_decay=.8, state_decay=.8)
        2d case
        >>> akf = AdaptiveExtendedKalmanFilter(curr_measurement_noise_cov=1e-5, curr_state = np.array([[1e-5], [0.]]), curr_state_estimate_cov=1e-7 * np.eye(2), curr_state_noise_cov=1e-9 * np.eye(2), measurement_decay=.8, state_decay=.8)
        """
        self.curr_state = _cast_float2mat(curr_state)
        self.curr_state_estimate_cov = _cast_float2mat(curr_state_estimate_cov)
        self.curr_measurement_noise_cov = _cast_float2mat(curr_measurement_noise_cov)
        self.curr_state_noise_cov = _cast_float2mat(curr_state_noise_cov)
        self.v = dof
        self.curr_prediction = np.nan
        self.gamma = gamma
        self.N_m = N_m
        self.epsilon = epsilon
        self.N_m_prime = N_m_prime
        self.epsilon_prime = epsilon_prime
        self.tau_k = tau_k
        self.lambda_k = lambda_k
        n,_ = curr_state_estimate_cov.shape
        if state_to_state_transition_mat is None:
            self.state_to_state_transition_mat = np.identity(n)
        else:
            #TODO : assert shape
            self.state_to_state_transition_mat = state_to_state_transition_mat

    def get_pred(self):
        return self.curr_prediction
    
    def get_state(self):
        return self.curr_state
    
    @staticmethod
    def kld_min_t_matching(mu1, Sigma1, v1, v2, N_m, epsilon):
        """
        :params mu1: mean vector 1
        :params Sigma1: covariance matrix 1
        :params v1: dof of distribution 1
        :params v2: dof of distribution 2
        :params Nm: maximum number of iterations
        :params tol: tolerance of convergency
        """
        n = mu1.shape[0]
        mu2 = mu1
        Sigma1_tilde = v1 / (v1 - 2) * Sigma1
        Sigma2_i0 = (v2 - 2) / v2 * Sigma1_tilde
        is_converged = False
        for _ in range(N_m):
            tr_tilde_Sigma1_mul_Sigma2_inv = np.trace(Sigma1_tilde @ np.linalg.inv(Sigma2_i0))
            Sigma2_i1 = (v2 + n) / (v1 + n) * ((v1 + tr_tilde_Sigma1_mul_Sigma2_inv) / (v2 + tr_tilde_Sigma1_mul_Sigma2_inv)) * Sigma1
            if np.linalg.norm(Sigma2_i1  - Sigma2_i0, 'fro') < epsilon:
                is_converged = True
                break
            Sigma2_i0 = Sigma2_i1
        Sigma2 = Sigma2_i1
        return mu2, Sigma2, v2, is_converged

    def fit_once(self, X: np.array, y: float):
        self.curr_state, self.curr_state_estimate_cov, self.curr_measurement_noise_cov, self.dof, *info= self._each_step(
            x_hat_k1k1=self.curr_state,
            P_k1k1=self.curr_state_estimate_cov, 
            F_k=self.state_to_state_transition_mat, 
            H_k=_cast_float2mat(X), 
            z_k=_cast_float2mat(y), 
            Q_tilde_k= self.curr_state_noise_cov, 
            R_tilde_k= self.curr_measurement_noise_cov, 
            tau_k= self.tau_k,
            lambda_k= self.lambda_k, 
            v= self.v, 
            N_m_prime=self.N_m_prime, 
            epsilon_prime=self.epsilon_prime, 
            N_m=self.N_m, 
            epsilon=self.epsilon)
        return info

    @staticmethod
    def _each_step(x_hat_k1k1, P_k1k1, F_k, H_k, z_k, Q_tilde_k, R_tilde_k,tau_k,lambda_k, v, N_m_prime, epsilon_prime, N_m, epsilon):
        """
        x_k1k1 = x_prev_posterior
        x_kk = x_curr_posterior
        x_kk1 = x_prior

        x_hat_k1k1: estimate of state vector at time k-1 of shape (n,1)
        P_k1k1: covariance matrix of state at time k-1 of shape (n,n)
        F_k: state transition matrix at time k, state_k1-to-state_k of shape (n,n)
        H_k: input-measurement matrix at time k, state_k1-to-measurement_k (prediction) of shape (m,n)
        z_k: output-measurement vector at time k of shape (m,1)
        Q_tilde_k: prior estimate of  covariance matrix, Q_k, of state noise, w_k, of shape (n,n)
        R_tilde_k: prior estimate of covariance matrix, R_k, of measurement noise, v_k, of shape (m.m)
        m, n: dimensions of measurement matrix H_k of shape (m, n)
        tau_k: correction weight for state estimate covariance matrix
        lambda_k: correction weight for measurement noise covariance matrix
        v: degree of freedom for state, state noise, measurement noise t-distribution (they are the same)
        N_m_prime: num of iters in one-step adaptive t-filtering algorithm
        epsilon_prime: convergence tolerance in one-step adaptive t-filtering algorithm
        N_m: num of iters in KLD minimization-based Student's t-matching algorithm
        epsilon: convergence tolerance in KLD minimization-based Student's t-matching algorithm
        """
        m, n = H_k.shape
        # TODO: assert the matrices are of the right size
        # time update
        x_hat_kk1 = F_k @ x_hat_k1k1  # might be F_k instead of F_k1
        P_tilde_kk1 = F_k @ P_k1k1 @ F_k.T + Q_tilde_k

        # measurement update
        # initialisation
        x_hat_kk_i0 = x_hat_kk1
        P_kk_i0 = P_tilde_kk1
        v_i0 = v
        is_modelling_converged = False
        for _ in range(N_m_prime):
            # first update
            A_k_i0 = v_i0/(v_i0 - 2) * P_kk_i0 + (x_hat_kk_i0 - x_hat_kk1) @ (x_hat_kk_i0 - x_hat_kk1).T
            P_kk1_i1 = (1-tau_k)*P_tilde_kk1 + tau_k*((v_i0-2)/v_i0)*A_k_i0

            # second update
            B_k_i0 = v_i0/(v_i0 - 2)*(H_k @ P_kk_i0 @ H_k.T) + (z_k - H_k @ x_hat_kk_i0) @ (z_k - H_k @ x_hat_kk_i0).T
            R_k_i1 = (1-lambda_k)*R_tilde_k + lambda_k*((v_i0-2)/v_i0)*B_k_i0

            # third update
            S_k_i1 = H_k @ P_kk1_i1 @ H_k.T + R_k_i1
            delta_k_i1 = np.sqrt((z_k - H_k @ x_hat_kk1).T @ np.linalg.inv(S_k_i1) @ (z_k - H_k @ x_hat_kk1))
            K_k_i1 = P_kk1_i1 @ H_k.T @ np.linalg.inv(S_k_i1)
            x_hat_kk_i1 = x_hat_kk1 + K_k_i1 @ (z_k - H_k @ x_hat_kk1)
            P_kk_i1 = (v + delta_k_i1**2)/(v + m) * (P_kk1_i1 - K_k_i1 @ H_k @ P_kk1_i1)
            v_i1 = v + m
            if np.linalg.norm(x_hat_kk_i1 - x_hat_kk_i0, 'fro')/np.linalg.norm(x_hat_kk_i0, 'fro') <= epsilon_prime:
                is_modelling_converged = True
                break
            x_hat_kk_i0 = x_hat_kk_i1
            P_kk_i0 = P_kk_i1
            v_i0 = v_i1
        x_hat_kk, P_kk, v, is_matching_converged = KLDMinimizationBasedAKF.kld_min_t_matching(x_hat_kk_i1, P_kk_i1, v_i1, v, N_m, epsilon)
        
        info = [x_hat_kk, P_kk, is_modelling_converged, is_matching_converged]
        return x_hat_kk, P_kk, R_k_i1, v, info

class UnkwonProbabilityLossBayesianKalmanFilter:
    def __init__(self):
        pass
