from typing import Tuple
import numpy as np

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter implementation for robot localization.
    Handles nonlinear dynamics and measurement models.

    State: [x, y, heading]
    Control: [forward_velocity, angular_velocity]
    """

    def __init__(self, env, initial_state=None):
        """Initialize EKF with environment and optional initial state."""
        self.env = env
        self.n_states = 3  # x, y, heading
        self.n_measurements = 1
        
        
        # Initialize state and covariance
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = np.array([5.0, 2.0, 0.0])  # Middle of environment
        
        # Initialize with larger uncertainty
        self.covariance = np.diag([0.5, 0.5, 0.2])  # x, y, heading variance
        
        # Motion model noise (state transition uncertainty)
        self.Q = np.diag([0.1, 0.1, 0.05])  # Reduced noise for smoother predictions
        
        # Measurement model noise
        # Use smaller noise for more confident measurements
        measurement_std = self.env.station_noise_std
        self.R = np.array([[measurement_std ** 2]])
        
        # Minimum and maximum variance constraints
        self.min_variance = np.array([0.01, 0.01, 0.01])  # Minimum allowed variance
        self.max_variance = np.array([2.0, 2.0, 1.0])  # Maximum allowed variance
        
        # Store history for debugging
        self.state_history = [self.state.copy()]
        self.covariance_history = [self.covariance.copy()]
        
        # Store last control inputs
        self.last_forward_vel = 0.0
        self.last_angular_vel = 0.0
        
        # Motion model noise parameters - Increased for more uncertainty
        self.motion_noise = np.diag([0.2, 0.2, 0.2])  # Position and heading noise
        
        # Update parameters
        self.max_position_update = 0.3  # Reduced for smoother updates

    def _motion_model(self, state: np.ndarray, forward_vel: float, angular_vel: float, dt: float) -> np.ndarray:
        """
        TODO: Implement the nonlinear motion model for the unicycle robot.

        This method should compute the predicted next state based on the control inputs
        (forward velocity and angular velocity), using the robot's kinematic model and
        terrain-dependent velocity scaling.

        Implementation steps:
        1. Extract x, y, and heading (theta) from the input state.
        2. Compute velocity scaling based on current position using:
             vel_scale = self.env._get_velocity_scaling(state[:2])
        3. Compute effective forward velocity:
             effective_vel = forward_vel * vel_scale
        4. Compute speed-dependent turning factor:
             - speed = |effective_vel| / self.env.max_velocity
             - speed_factor = np.clip(speed, 0.5, 1.5)
             - effective_angular_vel = angular_vel / speed_factor
        5. Compute next state:
             x_next = x + effective_vel * cos(theta) * dt
             y_next = y + effective_vel * sin(theta) * dt
             heading_next = (theta + effective_angular_vel * dt) % (2π)
        6. Return the predicted next state as a numpy array.

        Parameters:
            state : current state vector [x, y, heading]
            forward_vel : forward velocity input
            angular_vel : angular velocity input
            dt : time step duration
        """
        raise NotImplementedError("TODO: Implement the nonlinear motion model for the EKF.")

    def _motion_jacobian(self, state: np.ndarray, forward_vel: float, angular_vel: float, dt: float) -> np.ndarray:
        """
        TODO: Implement the Jacobian of the motion model (F = ∂f/∂x).

        This method should compute the Jacobian of the nonlinear motion model with
        respect to the state vector [x, y, heading]. You will likely want to compute
        the partial derivatives by hand and check them before implementation. 

        Implementation steps:
        1. Extract x, y, and heading (theta) from the input state.
        2. Compute the velocity scaling factor:
             vel_scale = self.env._get_velocity_scaling(state[:2])
           and its spatial derivatives:
             dvs_dx, dvs_dy = partial derivatives of velocity scaling with respect to x and y.
        3. Compute the effective forward velocity:
             effective_vel = forward_vel * vel_scale
           This term appears in the partial derivatives of x and y with respect to heading.
        4. Compute the speed-dependent turning factor:
             speed = |effective_vel| / self.env.max_velocity
             speed_factor = np.clip(speed, 0.5, 1.5)
             effective_angular_vel = angular_vel / speed_factor
           The derivative of heading (θ_next) will depend on this term.
        5. Build a 3×3 Jacobian matrix F initialized as identity (np.eye(3)).
           Then fill in its partial derivatives according to the unicycle model:
             - F[0,0], F[0,1] describe how x_next changes with x and y (through dvs_dx, dvs_dy).
             - F[0,2] = ∂x_next/∂θ ≈ -dt * effective_vel * sin(θ)
             - F[1,0], F[1,1] similarly affect y_next via sin(θ).
             - F[1,2] = ∂y_next/∂θ ≈ dt * effective_vel * cos(θ)
             - F[2,0], F[2,1] will include the speed_factor and dvs_dx, dvs_dy respectively.
             - F[2,2] = 1 (heading derivative with respect to itself).
        6. Return the resulting Jacobian matrix F (shape 3×3).

        Parameters:
            state : current state vector [x, y, heading]
            forward_vel : forward velocity input
            angular_vel : angular velocity input
            dt : time step duration
        """
        raise NotImplementedError("TODO: Implement the motion Jacobian (F = ∂f/∂x).")


    def _measurement_model(self, state: np.ndarray) -> np.ndarray:
        # z = range to the known station 
        r = self.env.range_to_station(state[:2])
        return np.array([r])



    def _measurement_jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        TODO: Implement the Jacobian of the measurement model (H = ∂h/∂x).

        This method computes how small changes in the state [x, y, heading]
        affect the expected range measurement to the fixed station.
        The measurement model is nonlinear:
            z = sqrt((x - x_s)^2 + (y - y_s)^2)

            (z, the measurement, is the same as r, the range between the robot and station)

        Implementation steps:
        1. Compute position differences between the robot and the station:
             dx = x - x_s
             dy = y - y_s

             (x_s and y_s can be found in self.env.station[0] and self.env.station[1])

        2. Compute the predicted range:
             r = sqrt(dx^2 + dy^2)
           This represents the expected distance to the station.

        3. Compute the partial derivatives of the range with respect to
           each state variable using the chain rule:
             ∂r/∂x = ?   (compute these by differentiating the expression for range w.r.t. x and y)
             ∂r/∂y = ?
             ∂r/∂θ = 0   (heading does not affect range)
        

        4. Handle the special case where r is very close to zero (to avoid
           division by zero). In that case, return a zero matrix. 1e-9 is a good threshold.

        5. Return the Jacobian H as a 1×3 matrix:
             H = [[ ∂r/∂x, ∂r/∂y, ∂r/∂θ ]]

        Parameters:
            state : current state vector [x, y, heading]

        Returns:
            H : 1×3 Jacobian matrix of the measurement model
        """
        raise NotImplementedError("TODO: Implement the measurement Jacobian (H = ∂h/∂x).")\


    def _constrain_covariance(self):
        """Apply minimum and maximum constraints to the covariance diagonal."""
        diag_idx = np.diag_indices_from(self.covariance)
        self.covariance[diag_idx] = np.clip(
            self.covariance[diag_idx],
            self.min_variance,
            self.max_variance
        )

    def _normalize_heading(self):
        """Normalize heading to [0, 2π)"""
        self.state[2] = self.state[2] % (2 * np.pi)

    def predict(self, forward_vel: float, angular_vel: float, dt: float):
        """
        TODO: Implement the Extended Kalman Filter prediction step.
        
        This method should implement the prediction step of the EKF using a nonlinear motion model
        that accounts for the robot's unicycle dynamics and terrain-dependent velocity scaling.
        
        Implementation steps:
        1. Store control inputs for later use:
           - self.last_forward_vel = action[0]
           - self.last_angular_vel = action[1]
           - self.last_dt = dt
        
        2. Update state using nonlinear motion model:
           - Call self._motion_model(self.state, forward_vel, angular_vel, dt) and 
             store result in self.state
           - This gives you the predicted next state
           - Call self._normalize_heading() to keep heading in [0, 2π]
        
        3. Compute Jacobian of motion model:
           - Call self._motion_jacobian(self.state, forward_vel, angular_vel, dt)
           - This gives you the linearized state transition matrix F
        
        4. Update covariance using the EKF covariance update equation:
           - P = F @ P @ F.T + Q
           - Use self.covariance for P and self.Q for process noise
        
        5. Apply covariance constraints:
           - Call self._constrain_covariance()
           - This ensures numerical stability
        
        6. Store state and covariance history:
           - Append copies of current state and covariance to self.state_history and self.covariance_history
        
        Parameters:
            forward_vel: forward velocity at time t
            angular_vel: angular velocity at time t
            dt (float): Time step duration
        """
        raise NotImplementedError("TODO: Implement the Extended Kalman Filter prediction step")

    def update(self, measurements: np.ndarray):
        """
        TODO: Implement the Extended Kalman Filter update step.
        
        This method should implement the update step of the EKF using nonlinear range measurements
        from multiple beams. The measurement model accounts for hallway geometry and obstacles.
        
        Implementation steps:
        1. Get expected measurements:
           - Reshape actual measurement received from the measurements input to (1,)
           - Call self._measurement_model(self.state) to get expected measurement
        
        2. Compute measurement Jacobian:
           - Call self._measurement_jacobian(self.state)
           - This linearizes the measurement model around current state
        
        3. Compute Kalman gain:
           - Innovation covariance: S = H @ P @ H.T + R
                - P and R are stored as attributes of self (self.covariance and self.R)
           - Kalman gain: K = P @ H.T @ inv(S)
        
        4. Update state:
           - Compute innovation: y = measurements - expected_z
           - Compute state update: dx = K @ y, use .reshape(-1) to change (3,1) vector to 
            (3,) for apply update step
           - Limit update magnitude if needed (based on self.max_position_update)
           - Apply update: self.state = self.state + dx
           - Call self._normalize_heading()
        
        5. Update covariance:
           - Covariance update: 
             P = (I - KH)P
           - Can use Joseph form for better numerical stability:
             P = (I - KH)P(I - KH).T + KRK.T
           - Call self._constrain_covariance()
        
        6. Store updated state and covariance:
           - Append to state to self.state_history and covariance to self.covariance_history
        
        Parameters:
            measurements (np.array) : single distance measurement to ranging station
        """
        raise NotImplementedError("TODO: Implement the Extended Kalman Filter update step")

    # --- Compatibility aliases (for visualizer & tests) ---
    @property
    def mu(self):
        return self.state

    @mu.setter
    def mu(self, v):
        self.state = np.asarray(v)

    @property
    def Sigma(self):
        return self.covariance

    @Sigma.setter
    def Sigma(self, M):
        self.covariance = np.asarray(M)

    def compute_motion_jacobian(self):
        """Wrapper for motion Jacobian (for test compatibility)."""
        return self._motion_jacobian(
            self.state,
            self.last_forward_vel,
            self.last_angular_vel,
            getattr(self, "last_dt", self.env.dt),
        )

    def compute_measurement_jacobian(self):
        """Wrapper for measurement Jacobian (for test compatibility)."""
        return self._measurement_jacobian(self.state)


    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current state estimate and covariance."""
        return self.state.copy(), self.covariance.copy()

    def get_position(self) -> np.ndarray:
        """Return current position estimate."""
        return self.state[:2].copy()

    def get_velocity(self) -> np.ndarray:
        """Return current velocity estimate based on control inputs."""
        heading = self.state[2]
        vel_scale = self.env._get_velocity_scaling(self.state[:2])
        effective_vel = self.last_forward_vel * vel_scale
        return np.array([
            effective_vel * np.cos(heading),
            effective_vel * np.sin(heading)
        ])

    def get_state_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the history of states and covariances."""
        return np.array(self.state_history), np.array(self.covariance_history)

    def reset(self, initial_state=None):
        """Reset the filter state."""
        if initial_state is not None:
            self.state = initial_state.copy()
        else:
            self.state = np.array([5.0, 2.0, 0.0])
        
        # Reset covariance to initial uncertainty
        self.covariance = np.diag([0.5, 0.5, 0.2])
        
        # Reset control inputs
        self.last_forward_vel = 0.0
        self.last_angular_vel = 0.0
        
        # Clear history
        self.state_history = [self.state.copy()]
        self.covariance_history = [self.covariance.copy()]
