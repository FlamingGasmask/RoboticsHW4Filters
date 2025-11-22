import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

from filtering_exercises_particle_filter.environments import MultiModalWorld
from filtering_exercises_particle_filter.assignment3_particle_filter.particle_filter import ParticleFilter

def plot_robot_and_particles(ax, true_state, particles, weights, landmarks, obstacles):
    """Plot the robot state, particles, and environment."""
    ax.clear()
    
    # Plot obstacles
    for obstacle in obstacles:
        ax.fill(*zip(*obstacle), color='gray', alpha=0.3)
    
    # Plot particles with weights as colors
    ax.scatter(particles[:, 0], particles[:, 1], 
              c=weights, cmap='viridis', 
              s=20, alpha=0.5, label='Particles')
    
    # Plot particle directions (for a subset of particles)
    n_arrows = min(20, len(particles))
    arrow_indices = np.random.choice(len(particles), n_arrows, p=weights/np.sum(weights))
    arrow_length = 0.3
    for idx in arrow_indices:
        dx = arrow_length * np.cos(particles[idx, 2])
        dy = arrow_length * np.sin(particles[idx, 2])
        ax.arrow(particles[idx, 0], particles[idx, 1], dx, dy,
                head_width=0.1, head_length=0.1, fc='k', ec='k', alpha=0.5)
    
    # Plot true robot state
    if true_state is not None:
        ax.plot(true_state[0], true_state[1], 'r*', markersize=15, label='True State')
        # Plot true heading
        dx = arrow_length * np.cos(true_state[2])
        dy = arrow_length * np.sin(true_state[2])
        ax.arrow(true_state[0], true_state[1], dx, dy,
                head_width=0.1, head_length=0.1, fc='r', ec='r')
    
    # Set plot properties
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Particle Filter Visualization')
    ax.legend()

def test_particle_filter_visualization():
    """Run a test visualization of the particle filter."""
    # Create environment and filter
    env = MultiModalWorld()
    pf = ParticleFilter(env, num_particles=100)
    
    # Setup plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Initialize true state
    env.agent_pos = np.array([5.0, 5.0])  # Start in middle
    env.agent_heading = 0.0  # Facing right
    true_state = np.array([env.agent_pos[0], env.agent_pos[1], env.agent_heading])
    
    try:
        # Run simulation
        for t in range(100):  # 10 seconds with dt=0.1
            # Alternate between straight and curved motion
            if t % 20 < 10:
                action = np.array([0.5, 0.0])  # Move straight
            else:
                action = np.array([0.3, 0.5])  # Turn
            
            # Get measurements
            z = env._get_sensor_reading()
            
            # Update particle filter
            pf.predict(action)
            pf.update(z)
            
            # Resample occasionally
            if t % 5 == 0:
                pf.resample()
            
            # Move true state
            next_state, _, _ = env.step(action)
            true_state = np.array([next_state[0], next_state[1], env.agent_heading])
            
            # Visualize
            plot_robot_and_particles(ax, true_state, pf.particles, pf.weights,
                                   [], env.obstacles)
            plt.draw()
            plt.pause(0.01)
            
            # Display time
            ax.text(0.02, 0.98, f'Time: {t*0.1:.1f}s',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top')
            
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    test_particle_filter_visualization() 