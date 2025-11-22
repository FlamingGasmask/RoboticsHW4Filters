from typing import List, Tuple

import numpy as np


class GridWorld:
    """
    A discrete grid world environment with obstacles and lidar-like sensing.
    The agent can move in 4 directions (up, right, down, left).
    """

    def __init__(
        self,
        size: Tuple[int, int] | int = (10, 10),
        obstacle_positions: List[Tuple[int, int]] = None,
        sensor_noise_std: float = 0.1,
    ):
        # Handle single integer size input
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.grid = np.zeros(self.size)  # 0: free, 1: obstacle

        # Set default obstacles if none provided
        if obstacle_positions is None:
            # Scale obstacle positions based on grid size
            if isinstance(size, int) and size < 10:
                # For smaller grids, use fewer obstacles in valid positions
                obstacle_positions = [(1, 1), (1, 2), (2, 2)]
            else:
                obstacle_positions = [(2, 2), (2, 3), (5, 5), (6, 5), (7, 5)]

        # Place obstacles (only if they're within bounds)
        for pos in obstacle_positions:
            if 0 <= pos[0] < self.size[0] and 0 <= pos[1] < self.size[1]:
                self.grid[pos] = 1

        # Agent state
        self.agent_pos = (0, 0)
        self.agent_heading = 0  # 0: right, 1: up, 2: left, 3: down

        # Sensor parameters
        self.n_beams = 4  # Fixed for grid world
        self.sensor_noise_std = sensor_noise_std
        self.max_beam_length = max(self.size[0], self.size[1])

    def reset(
        self, position: Tuple[int, int] = None
    ) -> Tuple[Tuple[int, int], np.ndarray]:
        """Reset the environment with optional starting position."""
        if position is None:
            # Find random free position
            while True:
                pos = (
                    np.random.randint(0, self.size[0]),
                    np.random.randint(0, self.size[1]),
                )
                if self.grid[pos] == 0:
                    break
            self.agent_pos = pos
        else:
            self.agent_pos = position

        self.agent_heading = np.random.randint(0, 4)
        return self.agent_pos, self._get_sensor_reading()

    def step(self, action: int) -> Tuple[Tuple[int, int], np.ndarray, bool]:
        """
        Take a step in the environment.
        action: 0: forward, 1: turn right, 2: turn left
        Returns: (new_position, sensor_readings, collision)
        """
        if action == 0:  # Move forward
            new_pos = self._get_forward_position()
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos
                collision = False
            else:
                collision = True
        elif action == 1:  # Turn right
            self.agent_heading = (self.agent_heading + 1) % 4
            collision = False
        elif action == 2:  # Turn left
            self.agent_heading = (self.agent_heading - 1) % 4
            collision = False

        return self.agent_pos, self._get_sensor_reading(), collision

    def get_measurements(self, position: Tuple[int, int]) -> np.ndarray:
        """Get sensor measurements from a given position."""
        # Temporarily store current state
        old_pos = self.agent_pos

        # Move to the position we want measurements from
        self.agent_pos = position

        # Get measurements
        measurements = self._get_sensor_reading()

        # Restore original state
        self.agent_pos = old_pos

        return measurements

    def _get_forward_position(self) -> Tuple[int, int]:
        """Calculate new position after moving forward."""
        moves = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # right, up, left, down
        move = moves[self.agent_heading]
        return (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (in bounds and not obstacle)."""
        if (
            0 <= pos[0] < self.size[0]
            and 0 <= pos[1] < self.size[1]
            and self.grid[pos] == 0
        ):
            return True
        return False

    def _get_sensor_reading(self) -> np.ndarray:
        """Get noisy readings from 4 lidar beams."""
        readings = np.zeros(self.n_beams)
        beam_directions = [(self.agent_heading + i) % 4 for i in range(self.n_beams)]

        for i, direction in enumerate(beam_directions):
            # Cast ray until hitting obstacle or boundary
            moves = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # right, up, left, down
            move = moves[direction]

            curr_pos = self.agent_pos
            distance = 0

            while True:
                curr_pos = (curr_pos[0] + move[0], curr_pos[1] + move[1])
                distance += 1

                # Check if out of bounds
                if not (
                    0 <= curr_pos[0] < self.size[0] and 0 <= curr_pos[1] < self.size[1]
                ):
                    break

                # Check if hit obstacle
                if self.grid[curr_pos] == 1:
                    break

                if distance >= self.max_beam_length:
                    break

            # Add noise to reading
            noise = np.random.normal(0, self.sensor_noise_std)
            readings[i] = distance + noise

        return readings
