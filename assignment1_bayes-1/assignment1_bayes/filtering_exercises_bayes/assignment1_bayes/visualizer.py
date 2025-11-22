import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import time

class BayesFilterVisualizer:
    def __init__(self, env, bayes_filter):
        """Initialize visualizer with environment and filter."""
        self.env = env
        self.filter = bayes_filter
        
        # Setup plots
        plt.ion()  # Enable interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle('Bayes Filter Visualization')
        
        # Setup belief plot
        self.belief_plot = None
        self.ax1.set_title('Belief Distribution')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        
        # Setup environment plot
        self.ax2.set_title('Environment')
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        self._plot_environment()
        
        plt.tight_layout()

    def _plot_environment(self):
        """Plot the environment with obstacles."""
        self.ax2.clear()
        self.ax2.set_title('Environment')
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        
        # Plot grid
        for i in range(self.env.size[0]):
            for j in range(self.env.size[1]):
                if self.env.grid[i, j] == 1:  # Obstacle
                    self.ax2.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, 
                                               facecolor='black'))
        
        self.ax2.grid(True)
        self.ax2.set_xlim(-0.5, self.env.size[0]-0.5)
        self.ax2.set_ylim(-0.5, self.env.size[1]-0.5)

    def update_plots(self, true_state=None):
        """Update visualization with current belief and true state."""
        # Update belief plot
        belief = self.filter.get_belief()
        if self.belief_plot is None:
            self.belief_plot = self.ax1.imshow(belief, origin='lower', 
                                             vmin=0, vmax=np.max(belief))
            self.fig.colorbar(self.belief_plot, ax=self.ax1)
        else:
            self.belief_plot.set_data(belief)
            self.belief_plot.set_clim(vmin=0, vmax=np.max(belief))
        
        # Plot true state if provided
        if true_state is not None:
            self.ax2.plot(true_state[1], true_state[0], 'r*', 
                         markersize=10, label='True State')
            self.ax2.legend()
        
        # Plot most likely state
        most_likely = self.filter.get_most_likely_state()
        self.ax2.plot(most_likely[1], most_likely[0], 'g*', 
                     markersize=10, label='Estimated State')
        self.ax2.legend()
        
        plt.pause(0.1)

    def run_visualization(self, n_steps=20):
        """Run a visualization episode."""
        # Start from a random free position
        true_state = [np.random.randint(0, self.env.size[0]),
                     np.random.randint(0, self.env.size[1])]
        while self.env.grid[true_state[0], true_state[1]] == 1:
            true_state = [np.random.randint(0, self.env.size[0]),
                         np.random.randint(0, self.env.size[1])]
        
        for _ in range(n_steps):
            # Generate measurements from true state
            readings = self.env._get_sensor_reading()
            
            # Update filter
            self.filter.update(readings)
            
            # Move true state (random walk)
            action = np.random.randint(0, 3)  # Random action (forward, turn right, turn left)
            _, readings, collision = self.env.step(action)
            if not collision:
                true_state = list(self.env.agent_pos)
            
            # Predict next state
            self.filter.predict(action)
            
            # Update visualization
            self.update_plots(true_state)
            time.sleep(0.5)  # Pause to see the update
        
        plt.ioff()
        plt.show() 