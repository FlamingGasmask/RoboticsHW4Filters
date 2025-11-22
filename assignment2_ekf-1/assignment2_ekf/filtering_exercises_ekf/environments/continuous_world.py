import numpy as np
from typing import Tuple, List
import matplotlib.path as mpath
import matplotlib.patches as mpatches


class ContinuousWorld:
    """
    A continuous 2D environment with obstacles and lidar-like sensing.
    The agent can move continuously in the environment.
    """

    def __init__(
        self,
        size: Tuple[float, float] = (10.0, 10.0),
        n_beams: int = 8,
        obstacles: List[List[Tuple[float, float]]] = None,
    ):
        self.size = size
        
        # Parameters for velocity scaling
        self.terrain_frequency = 2.0
        self.max_velocity = 2.0

        # Default obstacles if none provided (polygonal obstacles)
        if obstacles is None:
            self.obstacles = [
                [(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)],  # Square
                [(5.0, 5.0), (7.0, 5.0), (6.0, 7.0)],  # Triangle
            ]
        else:
            self.obstacles = obstacles

        # Convert obstacles to matplotlib paths for collision detection
        self.obstacle_paths = [mpath.Path(obstacle) for obstacle in self.obstacles]

        # Agent state
        self.agent_pos = np.array([0.0, 0.0])
        self.agent_heading = 0.0  # radians

        # Sensor parameters
        self.n_beams = n_beams
        self.sensor_noise_std = 0.05
        self.max_beam_length = np.sqrt(size[0] ** 2 + size[1] ** 2)

        # added by Scott
        self.station = np.array([8.0, 2.0])   # ranging station (change as you like)
        self.station_noise_std = 0.05         # meters, measurement noise
        self.landmarks = [self.station]       # so the visualizer marks it


    def reset(self, position: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Reset the environment with optional starting position."""
        if position is None:
            # Random position (with collision checking)
            while True:
                pos = np.random.uniform([0, 0], self.size)
                if not self._check_collision(pos):
                    break
            self.agent_pos = pos
        else:
            self.agent_pos = position

        self.agent_heading = np.random.uniform(0, 2 * np.pi)
        return self.agent_pos.copy(), self._get_sensor_reading()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Take a step in the environment.
        action: [forward_velocity, angular_velocity]
        Returns: (new_position, sensor_readings, collision)
        """
        dt = 0.1  # Time step

        # Update heading
        self.agent_heading += action[1] * dt
        self.agent_heading = self.agent_heading % (2 * np.pi)

        # Update position
        direction = np.array([np.cos(self.agent_heading), np.sin(self.agent_heading)])
        new_pos = self.agent_pos + direction * action[0] * dt

        # Check bounds and collision
        collision = False
        if (
            0 <= new_pos[0] <= self.size[0]
            and 0 <= new_pos[1] <= self.size[1]
            and not self._check_collision(new_pos)
        ):
            self.agent_pos = new_pos
        else:
            collision = True

        return self.agent_pos.copy(), self._get_sensor_reading(), collision

    def _check_collision(self, position: np.ndarray) -> bool:
        """Check if position collides with any obstacle."""
        point = position.reshape(1, 2)
        return any(path.contains_points(point) for path in self.obstacle_paths)

    def _get_sensor_reading(self) -> np.ndarray:
        """Get noisy readings from lidar beams."""
        readings = np.zeros(self.n_beams)
        angles = np.linspace(0, 2 * np.pi, self.n_beams, endpoint=False)
        angles = (angles + self.agent_heading) % (2 * np.pi)

        for i, angle in enumerate(angles):
            direction = np.array([np.cos(angle), np.sin(angle)])

            # Ray casting with small steps
            step_size = 0.1
            curr_pos = self.agent_pos.copy()
            distance = 0.0

            while distance < self.max_beam_length:
                curr_pos += direction * step_size
                distance += step_size

                # Check bounds
                if not (
                    0 <= curr_pos[0] <= self.size[0]
                    and 0 <= curr_pos[1] <= self.size[1]
                ):
                    break

                # Check obstacle collision
                if self._check_collision(curr_pos):
                    break

            # Add noise to reading
            noise = np.random.normal(0, self.sensor_noise_std)
            readings[i] = distance + noise

        return readings

    def _get_velocity_scaling(self, position: np.ndarray) -> float:
        """Get velocity scaling based on position in environment."""
        x, y = position
        # Create a terrain-dependent velocity scaling
        # Using a sinusoidal pattern that varies with position
        scale = 1.0 + 0.5 * np.sin(self.terrain_frequency * x) * np.sin(self.terrain_frequency * y)
        return np.clip(scale, 0.5, 1.5)  # Ensure scaling is between 0.5 and 1.5

    def range_to_station(self, pos: np.ndarray) -> float:
        """Exact Euclidean range from pos=[x,y,...] to the station (no noise)."""
        dx, dy = pos[0] - self.station[0], pos[1] - self.station[1]
        return float(np.hypot(dx, dy))

    def noisy_range_to_station(self, pos: np.ndarray) -> float:
        """Noisy range measurement to the station."""
        r = self.range_to_station(pos)
        return float(r + np.random.normal(0.0, self.station_noise_std))