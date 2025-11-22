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
        # 1. Extract x, y, and heading (theta) from the input state.
        x, y, heading = state

        # 2. Compute velocity scaling based on current position using:
        #      vel_scale = self.env._get_velocity_scaling(state[:2])
        vel_scale = self.env._get_velocity_scaling(state[:2])

        # 3. Compute effective forward velocity:
        #      effective_vel = forward_vel * vel_scale
        effective_vel = forward_vel * vel_scale

        # 4. Compute speed-dependent turning factor:
        #      - speed = |effective_vel| / self.env.max_velocity
        #      - speed_factor = np.clip(speed, 0.5, 1.5)
        #      - effective_angular_vel = angular_vel / speed_factor
        max_vel = getattr(self.env, "max_velocity", 1.0)
        if max_vel <= 0:
            speed = 0.0
        else:
            speed = abs(effective_vel) / max_vel
        speed_factor = np.clip(speed, 0.5, 1.5)
        effective_angular_vel = angular_vel / speed_factor

        # 5. Compute next state:
        #      x_next = x + effective_vel * cos(theta) * dt
        #      y_next = y + effective_vel * sin(theta) * dt
        #      heading_next = (theta + effective_angular_vel * dt) % (2π)
        x_next       = x + effective_vel * np.cos(heading) * dt
        y_next       = y + effective_vel * np.sin(heading) * dt
        heading_next = heading + effective_angular_vel * dt
        heading_next = heading_next % (2 * np.pi)

        # 6. Return the predicted next state as a numpy array.
        return np.array([x_next, y_next, heading_next])
        # raise NotImplementedError("TODO: Implement the nonlinear motion model for the EKF.")

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
        # 1. Extract x, y, and heading (theta) from the input state.
        x, y, heading = state
        eps = 1e-3
        
        # 2. Compute velocity scaling based on current position using:
        #      vel_scale = self.env._get_velocity_scaling(state[:2])
        vel_scale = self.env._get_velocity_scaling(state[:2])
        
        # 3. Compute effective forward velocity:
        #      effective_vel = forward_vel * vel_scale
        effective_vel = forward_vel * vel_scale
        
        # 4. Compute speed-dependent turning factor:
        #      - speed = |effective_vel| / self.env.max_velocity
        #      - speed_factor = np.clip(speed, 0.5, 1.5)
        #      - effective_angular_vel = angular_vel / speed_factor
        max_vel = getattr(self.env, "max_velocity", 1.0)
        if max_vel <= 0:
            speed = 0.0
        else:
            speed = abs(effective_vel) / max_vel
        speed_factor = np.clip(speed, 0.5, 1.5)
        effective_angular_vel = angular_vel / speed_factor
            
        # 5. Build a 3×3 Jacobian matrix F initialized as identity (np.eye(3)).
        #    Then fill in its partial derivatives according to the unicycle model:
        #      - F[0,0], F[0,1] describe how x_next changes with x and y (through dvs_dx, dvs_dy).
        #      - F[0,2] = ∂x_next/∂θ ≈ -dt * effective_vel * sin(θ)
        #      - F[1,0], F[1,1] similarly affect y_next via sin(θ).
        #      - F[1,2] = ∂y_next/∂θ ≈ dt * effective_vel * cos(θ)
        #      - F[2,0], F[2,1] will include the speed_factor and dvs_dx, dvs_dy respectively.
        #      - F[2,2] = 1 (heading derivative with respect to itself).
        F = np.eye(3, dtype=float)
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        pos = np.array([x, y], dtype=float)
        # Del x
        vs_x_plus  = self.env._get_velocity_scaling(pos + np.array([eps, 0.0]))
        vs_x_minus = self.env._get_velocity_scaling(pos - np.array([eps, 0.0]))
        dvs_dx = (vs_x_plus - vs_x_minus) / (2.0 * eps)

        # Del y
        vs_y_plus  = self.env._get_velocity_scaling(pos + np.array([0.0, eps]))
        vs_y_minus = self.env._get_velocity_scaling(pos - np.array([0.0, eps]))
        dvs_dy = (vs_y_plus - vs_y_minus) / (2.0 * eps)
        
        
        F[0, 0]  = 1 + forward_vel * dvs_dx * cos_h * dt
        F[0, 1]  = forward_vel * dvs_dy * cos_h * dt
        F[0, 2]  = -effective_vel * sin_h * dt

        F[1, 0]  = forward_vel * dvs_dx * sin_h * dt
        F[1, 1]  = 1 + forward_vel * dvs_dy * sin_h * dt
        F[1, 2]  = effective_vel * cos_h * dt
        
        denom = speed_factor * max_vel
        F[2, 0]  = -(effective_angular_vel * forward_vel * dvs_dx * dt) / denom
        F[2, 1]  = -(effective_angular_vel * forward_vel * dvs_dy * dt) / denom
        F[2,2]   = 1

        # 6. Return the resulting Jacobian matrix F (shape 3×3).
        return F
        # raise NotImplementedError("TODO: Implement the motion Jacobian (F = ∂f/∂x).")


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
        x, y, _ = state

        # Station position
        x_s, y_s = self.env.station[0], self.env.station[1]

        # 1. Compute position differences between the robot and the station:
        #      dx = x - x_s
        #      dy = y - y_s

        #      (x_s and y_s can be found in self.env.station[0] and self.env.station[1])
        dx = x - x_s
        dy = y - y_s

        # 2. Compute the predicted range:
        #      r = sqrt(dx^2 + dy^2)
        #    This represents the expected distance to the station.
        r = np.sqrt(dx**2 + dy**2)

        # 3. Compute the partial derivatives of the range with respect to
        #    each state variable using the chain rule:
        #      ∂r/∂x = ?   (compute these by differentiating the expression for range w.r.t. x and y)
        #      ∂r/∂y = ?
        #      ∂r/∂θ = 0   (heading does not affect range)
        dr_dx = dx / r
        dr_dy = dy / r
        dr_dtheta = 0.0  # heading does not affect range
        
        # 4. Handle the special case where r is very close to zero (to avoid
        #    division by zero). In that case, return a zero matrix. 1e-9 is a good threshold.
        if r < 1e-9:
            return np.zeros((1, 3))

        # 5. Return the Jacobian H as a 1×3 matrix:
        #      H = [[ ∂r/∂x, ∂r/∂y, ∂r/∂θ ]]
        H = np.array([[dr_dx, dr_dy, dr_dtheta]])
        return H
        # raise NotImplementedError("TODO: Implement the measurement Jacobian (H = ∂h/∂x).")\


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
        # 1. Store control inputs for later use:
        #    - self.last_forward_vel = action[0]
        #    - self.last_angular_vel = action[1]
        #    - self.last_dt = dt
        self.last_forward_vel = forward_vel
        self.last_angular_vel = angular_vel
        self.last_dt = dt

        # 2. Update state using nonlinear motion model:
        #    - Call self._motion_model(self.state, forward_vel, angular_vel, dt) and 
        #      store result in self.state
        #    - This gives you the predicted next state
        #    - Call self._normalize_heading() to keep heading in [0, 2π]
        self.state = self._motion_model(self.state, forward_vel, angular_vel, dt)
        self._normalize_heading()

        # 3. Compute Jacobian of motion model:
        #    - Call self._motion_jacobian(self.state, forward_vel, angular_vel, dt)
        #    - This gives you the linearized state transition matrix F
        F = self._motion_jacobian(self.state, forward_vel, angular_vel, dt)

        # 4. Update covariance using the EKF covariance update equation:
        #    - P = F @ P @ F.T + Q
        #    - Use self.covariance for P and self.Q for process noise
        self.covariance = F @ self.covariance @ F.T + self.Q

        # 5. Apply covariance constraints:
        #    - Call self._constrain_covariance()
        #    - This ensures numerical stability
        self._constrain_covariance()

        # 6. Store state and covariance history:
        #    - Append copies of current state and covariance to self.state_history and self.covariance_history
        self.state_history.append(self.state.copy())
        self.covariance_history.append(self.covariance.copy())
        # raise NotImplementedError("TODO: Implement the Extended Kalman Filter prediction step")

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
        # 1. Get expected measurements:
        #    - Reshape actual measurement received from the measurements input to (1,)
        #    - Call self._measurement_model(self.state) to get expected measurement
        z = np.asarray(measurements).reshape(1,)
        z_pred = self._measurement_model(self.state).reshape(1,)

        # 2. Compute measurement Jacobian:
        #    - Call self._measurement_jacobian(self.state)
        #    - This linearizes the measurement model around current state
        H = self._measurement_jacobian(self.state)  # shape (1,3)

        # 3. Compute Kalman gain:
        #    - Innovation covariance: S = H @ P @ H.T + R
        #         - P and R are stored as attributes of self (self.covariance and self.R)
        #    - Kalman gain: K = P @ H.T @ inv(S)
        P = self.covariance
        S = H @ P @ H.T + self.R  # (1,1)
        S_inv = np.linalg.inv(S)
        K = P @ H.T @ S_inv       # (3,1)

        # 4. Update state:
        #    - Compute innovation: y = measurements - expected_z
        #    - Compute state update: dx = K @ y, use .reshape(-1) to change (3,1) vector to 
        #     (3,) for apply update step
        #    - Limit update magnitude if needed (based on self.max_position_update)
        #    - Apply update: self.state = self.state + dx
        #    - Call self._normalize_heading()
        y = (z - z_pred).reshape(1, 1)   # innovation (1,1)
        dx = (K @ y).reshape(-1)        # (3,)

        pos_update_norm = np.linalg.norm(dx[:2])
        if pos_update_norm > self.max_position_update and pos_update_norm > 0:
            dx[:2] *= self.max_position_update / pos_update_norm

        self.state = self.state + dx
        self._normalize_heading()

        # 5. Update covariance:
        #    - Covariance update: 
        #      P = (I - KH)P
        #    - Can use Joseph form for better numerical stability:
        #      P = (I - KH)P(I - KH).T + KRK.T
        #    - Call self._constrain_covariance()
        I = np.eye(self.n_states)
        KH = K @ H                   # (3,3)
        self.covariance = (I - KH) @ P @ (I - KH).T + K @ self.R @ K.T
        self._constrain_covariance()

        # 6. Store updated state and covariance:
        #    - Append to state to self.state_history and covariance to self.covariance_history
        self.state_history.append(self.state.copy())
        self.covariance_history.append(self.covariance.copy())
        # raise NotImplementedError("TODO: Implement the Extended Kalman Filter update step")

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
