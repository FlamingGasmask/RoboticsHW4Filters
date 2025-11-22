from typing import Tuple

import numpy as np

from .continuous_world import ContinuousWorld


class NonlinearWorld(ContinuousWorld):
    """
    A 2D environment with nonlinear dynamics.
    The robot's motion model includes:
    - Nonlinear velocity that depends on position (simulating terrain effects)
    - Angular velocity that depends on current speed
    - Random position jumps (simulating wheel slip or terrain bumps)
    - State: [x, y, heading]
    """

    def __init__(self):
        super().__init__()
        # Additional parameters for nonlinear dynamics
        self.max_velocity = 1.0
        self.min_velocity = 0.1
        self.terrain_frequency = 0.5  # Frequency of terrain variation
        self.dt = 0.1  # Time step for dynamics

        # Jump parameters
        self.jump_probability = 0.1  # Probability of a jump per step
        self.max_jump_distance = 0.2  # Maximum jump distance
        self.jump_noise_std = 0.05  # Standard deviation of jump noise

    def reset(self):
        """Reset the environment."""
        super().reset()
        # Start with same initial conditions as parent
        self.agent_heading = -np.pi / 4  # Point towards bottom-left

    def _get_velocity_scaling(self, position: np.ndarray) -> float:
        """
        Get velocity scaling based on position (simulates terrain effects).
        Creates a sinusoidal pattern of fast/slow regions.
        """
        x, y = position
        # Create a 2D sinusoidal pattern
        terrain = np.sin(self.terrain_frequency * x) * np.cos(
            self.terrain_frequency * y
        )
        # Map to range [0.5, 1.5] to scale velocity
        scaling = 1.0 + 0.5 * terrain
        return scaling

    def _apply_random_jump(self, position: np.ndarray) -> np.ndarray:
        """Apply a random jump to the position (simulates wheel slip)."""
        if np.random.random() < self.jump_probability:
            # Generate random jump direction
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, self.max_jump_distance)

            # Add some noise to the jump
            jump = np.array(
                [
                    np.cos(angle) * distance + np.random.normal(0, self.jump_noise_std),
                    np.sin(angle) * distance + np.random.normal(0, self.jump_noise_std),
                ]
            )

            return position + jump
        return position

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Take a step in the environment with nonlinear dynamics.
        Args:
            action: [forward_velocity, angular_velocity]
        Returns:
            state: [x, y, heading]
            measurements: Array of beam distances
            collision: Whether collision occurred
        """
        forward_vel, angular_vel = action

        # Apply nonlinear velocity scaling based on position
        vel_scale = self._get_velocity_scaling(self.agent_pos)
        effective_vel = forward_vel * vel_scale

        # Make angular velocity depend on speed
        # (harder to turn at higher speeds)
        speed = np.abs(effective_vel) / self.max_velocity
        speed_factor = np.clip(speed, 0.5, 1.5)
        effective_angular_vel = angular_vel / speed_factor

        # Update heading with nonlinear angular velocity
        new_heading = self.agent_heading + effective_angular_vel * self.dt
        self.agent_heading = new_heading % (2 * np.pi)

        # Compute velocities in global frame with nonlinear effects
        vx = effective_vel * np.cos(self.agent_heading)
        vy = effective_vel * np.sin(self.agent_heading)

        # Update position
        new_pos = self.agent_pos + np.array([vx, vy]) * self.dt

        # Apply random jump (simulating wheel slip or terrain effects)
        new_pos = self._apply_random_jump(new_pos)

        # Check for collisions and update position
        if self._check_collision(new_pos):
            state = np.array([self.agent_pos[0], self.agent_pos[1], self.agent_heading])
            return state, self._get_sensor_reading(), True

        self.agent_pos = new_pos
        state = np.array([self.agent_pos[0], self.agent_pos[1], self.agent_heading])
        return state, self._get_sensor_reading(), False
