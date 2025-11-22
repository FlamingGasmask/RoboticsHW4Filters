import time

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from filtering_exercises_bayes.assignment1_bayes.bayes_filter import BayesFilter
from filtering_exercises_bayes.environments.grid_world import GridWorld


class BayesFilterVisualizer:
    def __init__(self, env, bayes_filter):
        """Initialize visualizer with environment and filter."""
        self.env = env
        self.filter = bayes_filter

        # Setup plots
        plt.ion()  # Enable interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle("Bayes Filter Visualization")

        # Setup belief plot
        self.belief_plot = None
        self.ax1.set_title("Belief Distribution")
        self.ax1.set_xlabel("X")
        self.ax1.set_ylabel("Y")

        # Setup environment plot
        self.ax2.set_title("Environment")
        self.ax2.set_xlabel("X")
        self.ax2.set_ylabel("Y")
        self._plot_environment()

        plt.tight_layout()

    def _plot_environment(self):
        """Plot the environment with obstacles."""
        self.ax2.clear()
        self.ax2.set_title("Environment")
        self.ax2.set_xlabel("X")
        self.ax2.set_ylabel("Y")

        # Plot grid
        for i in range(self.env.size[0]):
            for j in range(self.env.size[1]):
                if self.env.grid[i, j] == 1:  # Obstacle
                    self.ax2.add_patch(
                        Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="black")
                    )

        self.ax2.grid(True)
        self.ax2.set_xlim(-0.5, self.env.size[0] - 0.5)
        self.ax2.set_ylim(-0.5, self.env.size[1] - 0.5)

    def update_plots(self, true_state=None):
        """Update visualization with current belief and true state."""
        # Update belief plot
        belief = self.filter.get_belief()
        if self.belief_plot is None:
            self.belief_plot = self.ax1.imshow(
                belief, origin="lower", vmin=0, vmax=np.max(belief)
            )
            self.fig.colorbar(self.belief_plot, ax=self.ax1)
        else:
            self.belief_plot.set_data(belief)
            self.belief_plot.set_clim(vmin=0, vmax=np.max(belief))

        # Clear environment plot and redraw it
        self.ax2.clear()
        self._plot_environment()

        # Plot true state if provided
        if true_state is not None:
            self.ax2.plot(
                true_state[1], true_state[0], "r*", markersize=10, label="True State"
            )

        # Plot most likely state
        most_likely = self.filter.get_most_likely_state()
        self.ax2.plot(
            most_likely[1], most_likely[0], "g*", markersize=10, label="Estimated State"
        )
        
        # Add legend once
        self.ax2.legend()

        plt.pause(0.1)


def test_bayes_filter_visualization():
    """Run a visualization episode of the Bayes filter."""
    # Create environment and filter
    obstacles = [(2, 2), (2, 3), (2, 4), (5, 5), (6, 5), (6, 6)]
    env = GridWorld(size=(10, 10), obstacle_positions=obstacles, sensor_noise_std=0.3)
    bayes_filter = BayesFilter(env)

    # Create visualizer
    viz = BayesFilterVisualizer(env, bayes_filter)

    # Run simulation
    true_state, _ = env.reset()  # Example true state
    n_steps = 30

    for _ in range(n_steps):
        # Generate measurements from true state
        measurements = env.get_measurements(true_state)

        # Update filter
        bayes_filter.update(measurements)

        # Move true state (example: random walk)
        action = np.random.randint(0, 3)  # Random action (0: forward, 1: turn right, 2: turn left)
        true_state, _, _ = env.step(action)
        true_state = list(true_state)  # Convert tuple to list to match existing code

        # Predict next state
        bayes_filter.predict(action)

        # Update visualization
        viz.update_plots(true_state)
        time.sleep(0.5)  # Pause to see the update

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    test_bayes_filter_visualization()
