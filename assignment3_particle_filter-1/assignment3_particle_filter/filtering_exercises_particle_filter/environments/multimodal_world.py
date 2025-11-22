from typing import List, Tuple

import numpy as np

from .nonlinear_world import NonlinearWorld


class MultiModalWorld(NonlinearWorld):
    """
    A 2D environment with multiple similar hallways that can create
    ambiguous measurements, leading to multi-modal beliefs.
    Inherits nonlinear dynamics from NonlinearWorld.
    """

    def __init__(self):
        super().__init__()

        # Create two identical hallways
        self.hallway_width = 2.0
        self.hallway_length = 8.0
        self.hallway_spacing = 5.0  # Space between hallways

        # Override default obstacles with hallway walls
        self.obstacles = self._create_hallways()

        # Convert obstacles to matplotlib paths for collision detection
        self._update_obstacle_paths()

        # Increase beam count for better hallway detection
        self.n_beams = 16

        # Adjust sensor parameters
        self.sensor_noise_std = 0.1  # Increased noise

        # Modify terrain parameters
        self.terrain_frequency = 0.8  # Higher frequency terrain variations
        self.jump_probability = 0.05  # Reduced jump probability

    def _create_hallways(self) -> List[List[Tuple[float, float]]]:
        """Create two identical parallel hallways."""
        obstacles = []

        # First hallway (bottom)
        y_base = 1.0
        obstacles.extend(
            [
                # Bottom wall
                [
                    (1.0, y_base),
                    (9.0, y_base),
                    (9.0, y_base + 0.2),
                    (1.0, y_base + 0.2),
                ],
                # Top wall
                [
                    (1.0, y_base + self.hallway_width),
                    (9.0, y_base + self.hallway_width),
                    (9.0, y_base + self.hallway_width + 0.2),
                    (1.0, y_base + self.hallway_width + 0.2),
                ],
            ]
        )

        # Second hallway (top)
        y_base = y_base + self.hallway_spacing
        obstacles.extend(
            [
                # Bottom wall
                [
                    (1.0, y_base),
                    (9.0, y_base),
                    (9.0, y_base + 0.2),
                    (1.0, y_base + 0.2),
                ],
                # Top wall
                [
                    (1.0, y_base + self.hallway_width),
                    (9.0, y_base + self.hallway_width),
                    (9.0, y_base + self.hallway_width + 0.2),
                    (1.0, y_base + self.hallway_width + 0.2),
                ],
            ]
        )

        # Add some distinguishing features (small obstacles) in each hallway
        # but keep them similar to maintain ambiguity
        obstacles.extend(
            [
                # Bottom hallway features
                [
                    (3.0, y_base - self.hallway_spacing + 0.5),
                    (3.2, y_base - self.hallway_spacing + 0.5),
                    (3.2, y_base - self.hallway_spacing + 0.7),
                    (3.0, y_base - self.hallway_spacing + 0.7),
                ],
                [
                    (7.0, y_base - self.hallway_spacing + 1.5),
                    (7.2, y_base - self.hallway_spacing + 1.5),
                    (7.2, y_base - self.hallway_spacing + 1.7),
                    (7.0, y_base - self.hallway_spacing + 1.7),
                ],
                # Top hallway features (similar but slightly different)
                [
                    (3.2, y_base + 0.5),
                    (3.4, y_base + 0.5),
                    (3.4, y_base + 0.7),
                    (3.2, y_base + 0.7),
                ],
                [
                    (7.2, y_base + 1.5),
                    (7.4, y_base + 1.5),
                    (7.4, y_base + 1.7),
                    (7.2, y_base + 1.7),
                ],
            ]
        )

        return obstacles

    def _update_obstacle_paths(self):
        """Update matplotlib paths after changing obstacles."""
        import matplotlib.path as mpath

        self.obstacle_paths = [mpath.Path(obstacle) for obstacle in self.obstacles]

    def reset(self, position: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Reset the environment, randomly choosing one of the hallways."""
        if position is None:
            # Randomly choose between bottom and top hallway
            hallway = np.random.choice([0, 1])
            y_base = 1.0 + (self.hallway_spacing if hallway else 0)

            # Random x position in the hallway
            x = np.random.uniform(1.5, 8.5)
            y = np.random.uniform(y_base + 0.3, y_base + self.hallway_width - 0.3)

            self.agent_pos = np.array([x, y])
        else:
            self.agent_pos = position

        self.agent_heading = np.random.uniform(
            -np.pi / 4, np.pi / 4
        )  # Face mostly forward
        return self.agent_pos.copy(), self._get_sensor_reading()
