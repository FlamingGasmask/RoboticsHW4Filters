from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Arrow, Circle, Polygon

from filtering_exercises.environments.grid_world import GridWorld


class Visualizer:
    """Visualizer for both GridWorld and ContinuousWorld environments."""

    def __init__(self, env, episode_length: int = 100, show_beams: bool = True):
        self.env = env
        self.episode_length = episode_length
        self.show_beams = show_beams
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        # Initialize animation-related attributes
        self.agent_marker = None
        self.beam_lines = []
        self.step_count = 0

    def visualize_episode(self, policy=None):
        """Run and visualize a full episode."""
        self.step_count = 0
        self.current_state, self.current_readings = self.env.reset()

        # Setup the plot
        self._setup_plot()

        def animate(frame):
            if policy is None:
                # Random actions if no policy provided
                if isinstance(self.env, GridWorld):
                    action = np.random.randint(0, 3)  # Forward, Right, Left
                else:
                    action = np.array([0.5, 0.2])  # Forward velocity, angular velocity
            else:
                action = policy(self.current_state, self.current_readings)

            self.current_state, self.current_readings, _ = self.env.step(action)
            self._update_plot(self.current_state, self.current_readings)
            self.step_count += 1

            return (self.agent_marker,) + tuple(self.beam_lines)

        anim = animation.FuncAnimation(
            self.fig, animate, frames=self.episode_length, interval=100, blit=True
        )
        plt.show()

    def _setup_plot(self):
        """Setup the initial plot."""
        self.ax.clear()

        if isinstance(self.env, GridWorld):
            self._setup_grid_world()
        else:
            self._setup_continuous_world()

        # Set axis properties
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        plt.tight_layout()

    def _setup_grid_world(self):
        """Setup visualization for GridWorld."""
        self.ax.set_xlim(-0.5, self.env.size[0] - 0.5)
        self.ax.set_ylim(-0.5, self.env.size[1] - 0.5)

        # Draw grid
        for i in range(self.env.size[0]):
            for j in range(self.env.size[1]):
                if self.env.grid[i, j] == 1:  # Obstacle
                    self.ax.add_patch(
                        plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="gray")
                    )

        # Initialize agent marker (smaller radius for better grid centering)
        self.agent_marker = Circle((0, 0), 0.25, color="blue")
        self.ax.add_patch(self.agent_marker)

        # Initialize beam lines
        self.beam_lines = []
        if self.show_beams:
            for _ in range(self.env.n_beams):
                (line,) = self.ax.plot([], [], "r-", alpha=0.5)
                self.beam_lines.append(line)

    def _setup_continuous_world(self):
        """Setup visualization for ContinuousWorld."""
        self.ax.set_xlim(0, self.env.size[0])
        self.ax.set_ylim(0, self.env.size[1])

        # Draw obstacles
        for obstacle in self.env.obstacles:
            self.ax.add_patch(Polygon(obstacle, facecolor="gray"))

        # Initialize agent marker
        self.agent_marker = Circle((0, 0), 0.2, color="blue")
        self.ax.add_patch(self.agent_marker)

        # Initialize beam lines
        self.beam_lines = []
        if self.show_beams:
            for _ in range(self.env.n_beams):
                (line,) = self.ax.plot([], [], "r-", alpha=0.5)
                self.beam_lines.append(line)

    def _update_plot(self, state, readings):
        """Update the plot with new state and sensor readings."""
        if isinstance(self.env, GridWorld):
            self._update_grid_world(state, readings)
        else:
            self._update_continuous_world(state, readings)

    def _update_grid_world(self, state, readings):
        """Update visualization for GridWorld."""
        # Convert from grid coordinates (i,j) to plot coordinates (x,y)
        self.agent_marker.center = (state[1], state[0])

        # Update beam lines if enabled
        if self.show_beams:
            moves = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # right, up, left, down
            beam_directions = [
                (self.env.agent_heading + i) % 4 for i in range(self.env.n_beams)
            ]

            for line, direction, reading in zip(
                self.beam_lines, beam_directions, readings
            ):
                move = moves[direction]
                end_x = state[1] + move[1] * reading
                end_y = state[0] + move[0] * reading
                line.set_data([state[1], end_x], [state[0], end_y])

    def _update_continuous_world(self, state, readings):
        """Update visualization for ContinuousWorld."""
        self.agent_marker.center = (state[0], state[1])

        # Update beam lines if enabled
        if self.show_beams:
            angles = np.linspace(0, 2 * np.pi, self.env.n_beams, endpoint=False)
            angles = (angles + self.env.agent_heading) % (2 * np.pi)

            for line, angle, reading in zip(self.beam_lines, angles, readings):
                end_x = state[0] + reading * np.cos(angle)
                end_y = state[1] + reading * np.sin(angle)
                line.set_data([state[0], end_x], [state[1], end_y])
