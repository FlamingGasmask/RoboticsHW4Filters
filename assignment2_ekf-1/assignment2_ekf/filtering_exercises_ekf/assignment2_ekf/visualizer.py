import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Circle, Polygon
import time

class EKFVisualizer:
    def __init__(self, env, filter):
        """Initialize visualizer with environment and filter."""
        self.env = env
        self.filter = filter
        
        # Create figure and axis
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Initialize plot objects
        self.true_pos = None
        self.est_pos = None
        self.true_heading = None
        self.est_heading = None
        self.uncertainty_ellipse = None
        
        # Setup initial plot
        self._setup_plot()
        self.true_range_circle = None
        self.est_range_circle = None
        plt.show()

    def _setup_plot(self):
        """Setup the plot with environment."""
        self.ax.clear()
        self.ax.set_title('Robot Localization')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        
        # Draw environment boundaries
        self.ax.set_xlim(-0.5, self.env.size[0] + 0.5)  # Add margins
        self.ax.set_ylim(-0.5, self.env.size[1] + 0.5)
        
        # Draw boundaries
        self.ax.plot([0, self.env.size[0], self.env.size[0], 0, 0], 
                    [0, 0, self.env.size[1], self.env.size[1], 0], 'k-', linewidth=2)
        
        # Draw obstacles
        for obstacle in self.env.obstacles:
            self.ax.add_patch(Polygon(obstacle, facecolor='gray', alpha=0.5))
        
        # Draw landmarks if they exist
        if hasattr(self.env, 'landmarks'):
            for i, landmark in enumerate(self.env.landmarks):
                self.ax.plot(landmark[0], landmark[1], 'k^', markersize=10, 
                           label=f'Landmark {i+1}')
        
        self.ax.grid(True)
        self.ax.set_aspect('equal')

    def _draw_uncertainty_ellipse(self, mean, covariance):
        """Draw uncertainty ellipse based on state covariance."""
        if self.uncertainty_ellipse is not None:
            self.uncertainty_ellipse.set_visible(False)

        # Extract position mean and covariance
        pos_mean = mean[:2]
        pos_cov = covariance[:2, :2]

        # Calculate ellipse parameters
        eigenvals, eigenvecs = np.linalg.eig(pos_cov)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 2 * np.sqrt(5.991 * eigenvals)  # 95% confidence interval

        # Create and add ellipse
        self.uncertainty_ellipse = Ellipse(xy=pos_mean, width=width, height=height, angle=angle,
                                         facecolor='none', edgecolor='g', alpha=0.5)
        self.ax.add_patch(self.uncertainty_ellipse)

    def update_plots(self, true_state=None):
        """Update visualization with current state estimates."""
        # Get current state estimate and covariance
        mean = self.filter.mu
        covariance = self.filter.Sigma
        
        # Draw heading for estimated position
        heading_len = 0.5
        dx = heading_len * np.cos(mean[2])
        dy = heading_len * np.sin(mean[2])
        if hasattr(self, 'est_heading') and self.est_heading is not None:
            self.est_heading.set_visible(False)
        self.est_heading = self.ax.arrow(mean[0], mean[1], dx, dy, 
                                       head_width=0.1, head_length=0.2, 
                                       fc='b', ec='b', alpha=0.7)
        
        # Update estimated position
        if self.est_pos is None:
            self.est_pos = Circle((mean[0], mean[1]), 0.2, color='b', 
                                alpha=0.5, label='Estimated Position')
            self.ax.add_patch(self.est_pos)
        else:
            self.est_pos.center = (mean[0], mean[1])
        
        # Draw uncertainty ellipse
        self._draw_uncertainty_ellipse(mean, covariance)
        
        # Update true position if provided
        if true_state is not None:
            # Draw heading for true position
            dx = heading_len * np.cos(true_state[2])
            dy = heading_len * np.sin(true_state[2])
            if hasattr(self, 'true_heading') and self.true_heading is not None:
                self.true_heading.set_visible(False)
            self.true_heading = self.ax.arrow(true_state[0], true_state[1], dx, dy, 
                                            head_width=0.1, head_length=0.2, 
                                            fc='r', ec='r', alpha=0.7)
            
            if self.true_pos is None:
                self.true_pos = Circle((true_state[0], true_state[1]), 0.2, 
                                     color='r', alpha=0.5, label='True Position')
                self.ax.add_patch(self.true_pos)
            else:
                self.true_pos.center = (true_state[0], true_state[1])
        
        # Draw/update predicted range circle (from EKF state)
        r_pred = self.env.range_to_station(self.filter.mu[:2])  # predicted r from current est
        if self.est_range_circle is not None:
            self.est_range_circle.remove()
        self.est_range_circle = Circle(self.env.station, r_pred, fill=False, ec='b', ls='--', alpha=0.4)
        self.ax.add_patch(self.est_range_circle)

        # Update legend
        handles = []
        labels = []
        if self.true_pos is not None:
            handles.extend([self.true_pos])
            labels.extend(['True Position'])
        if self.est_pos is not None:
            handles.extend([self.est_pos])
            labels.extend(['Estimated Position'])
        if hasattr(self.env, 'landmarks'):
            handles.extend([plt.Line2D([0], [0], marker='^', color='k', linestyle='None')])
            labels.extend(['Landmarks'])
        
        self.ax.legend(handles, labels, loc='upper right')
        plt.draw()
        plt.pause(0.01)  # Add a small pause to allow the plot to update

    def run_visualization(self, n_steps=50):
        """Run a visualization episode."""
        # Start from a random position
        true_state = np.array([
            np.random.uniform(1, self.env.size[0]-1),
            np.random.uniform(1, self.env.size[1]-1),
            np.random.uniform(-np.pi, np.pi)
        ])
        
        for _ in range(n_steps):
            # Generate random control input
            action = np.array([
                np.random.uniform(0.1, 0.5),  # Forward velocity
                np.random.uniform(-0.3, 0.3)   # Angular velocity
            ])
            
            # Get measurements
            z = self.env.get_measurements(true_state)
            
            # Update filter
            self.filter.predict(action)
            self.filter.update(z)
            
            # Move true state
            true_state = self.env.get_next_state(true_state, action)
            
            # Update visualization
            self.update_plots(true_state)
            time.sleep(0.1)
        
        plt.ioff()
        plt.show() 