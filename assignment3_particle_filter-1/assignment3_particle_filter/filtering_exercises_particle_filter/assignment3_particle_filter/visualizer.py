import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon
import time

class ParticleFilterVisualizer:
    def __init__(self, env, particle_filter):
        """Initialize visualizer with environment and filter."""
        self.env = env
        self.filter = particle_filter
        
        # Setup plot
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.suptitle('Particle Filter Visualization')
        
        # Initialize visualization elements
        self.particle_scatter = None
        self.true_pos = None
        self.est_pos = None
        self._setup_plot()
        
        plt.tight_layout()

    def _setup_plot(self):
        """Setup the plot with environment."""
        self.ax.clear()
        self.ax.set_title('Robot Localization')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        
        # Draw environment boundaries
        self.ax.set_xlim(0, self.env.size[0])
        self.ax.set_ylim(0, self.env.size[1])
        
        # Draw obstacles
        for obstacle in self.env.obstacles:
            self.ax.add_patch(Polygon(obstacle, facecolor='gray', alpha=0.3))
        
        # Draw landmarks
        for i, landmark in enumerate(self.env.landmarks):
            self.ax.plot(landmark[0], landmark[1], 'k^', markersize=10, 
                        label=f'Landmark {i+1}' if i == 0 else "")
        
        self.ax.grid(True)

    def update_plots(self, true_state=None):
        """Update visualization with current particle distribution."""
        # Get current particles and weights
        particles = self.filter.particles
        weights = self.filter.weights
        
        # Update particle scatter plot
        if self.particle_scatter is not None:
            self.particle_scatter.remove()
        self.particle_scatter = self.ax.scatter(particles[:, 0], particles[:, 1],
                                              c=weights, cmap='viridis', 
                                              alpha=0.5, label='Particles')
        
        # Get and plot estimated state
        est_state = self.filter.estimate_state()
        if self.est_pos is None:
            self.est_pos = Circle((est_state[0], est_state[1]), 0.2, 
                                color='b', label='Estimated Position')
            self.ax.add_patch(self.est_pos)
        else:
            self.est_pos.center = (est_state[0], est_state[1])
        
        # Update true position if provided
        if true_state is not None:
            if self.true_pos is None:
                self.true_pos = Circle((true_state[0], true_state[1]), 0.2,
                                     color='r', label='True Position')
                self.ax.add_patch(self.true_pos)
            else:
                self.true_pos.center = (true_state[0], true_state[1])
        
        # Draw particle directions (for a subset of particles)
        n_arrows = min(20, len(particles))
        arrow_indices = np.random.choice(len(particles), n_arrows, p=weights/np.sum(weights))
        arrow_length = 0.3
        for idx in arrow_indices:
            dx = arrow_length * np.cos(particles[idx, 2])
            dy = arrow_length * np.sin(particles[idx, 2])
            self.ax.arrow(particles[idx, 0], particles[idx, 1], dx, dy,
                         head_width=0.1, head_length=0.1, fc='k', ec='k', alpha=0.5)
        
        # Update legend and colorbar
        self.ax.legend()
        plt.colorbar(self.particle_scatter, label='Particle Weights')
        plt.pause(0.1)

    def run_visualization(self, n_steps=50):
        """Run a visualization episode."""
        # Start from a random position
        true_state = np.array([
            np.random.uniform(1, self.env.size[0]-1),
            np.random.uniform(1, self.env.size[1]-1),
            np.random.uniform(-np.pi, np.pi)
        ])
        
        try:
            for t in range(n_steps):
                # Generate random control input
                if t % 20 < 10:
                    action = np.array([0.5, 0.0])  # Move straight
                else:
                    action = np.array([0.3, 0.5])  # Turn
                
                # Get measurements
                z = self.env.get_measurements(true_state)
                
                # Update filter
                self.filter.predict(action)
                self.filter.update(z)
                
                # Resample occasionally
                if t % 5 == 0:
                    self.filter.resample()
                
                # Move true state
                true_state = self.env.get_next_state(true_state, action)
                
                # Update visualization
                self.update_plots(true_state)
                
                # Display time
                self.ax.text(0.02, 0.98, f'Time: {t*0.1:.1f}s',
                           transform=self.ax.transAxes, fontsize=10,
                           verticalalignment='top')
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nVisualization interrupted by user")
        
        plt.ioff()
        plt.show() 