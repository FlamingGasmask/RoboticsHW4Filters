#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import time

from filtering_exercises_ekf.environments import NonlinearWorld
from filtering_exercises_ekf.assignment2_ekf.extended_kalman_filter import ExtendedKalmanFilter
from filtering_exercises_ekf.assignment2_ekf.visualizer import EKFVisualizer

def test_ekf_visualization():
    """Run a test visualization of the EKF."""
    # Create environment and filter
    env = NonlinearWorld()
    print(f"\nEnvironment Configuration:")
    print(f"Ranging station located at: {env.station if hasattr(env, 'station') else '(not defined yet)'}")
    print(f"Environment size: {env.size}")
    
    # Set initial state away from obstacles and boundaries
    initial_state = np.array([7.0, 7.0, -np.pi/4])
    env.reset()
    env.agent_pos = initial_state[:2]
    env.agent_heading = initial_state[2]
    
    # Initialize EKF with some uncertainty about initial state
    initial_state_ekf = initial_state + np.random.normal(0, [0.2, 0.2, 0.1])
    ekf = ExtendedKalmanFilter(env, initial_state_ekf)
    
    # Create visualizer
    vis = EKFVisualizer(env, ekf)
    
    # Initialize error tracking with timestamps
    position_errors = []
    heading_errors = []
    times = []
    measurement_innovations = []
    velocity_scales = []
    
    

    
    # Add legend
    vis.ax.legend()
    
    
    # Simulation loop
    try:
        for i in range(200):
            t = i * env.dt
            times.append(t)
            
            vel_scale = env._get_velocity_scaling(ekf.mu[:2])
            velocity_scales.append(vel_scale)
            
            # Generate control input - smoother motion for better tracking
            if i < 50:  # Move diagonally down-left
                forward_vel = 0.3
                angular_vel = 0.0
                if (env.agent_pos[0] < 1.0 or env.agent_pos[0] > env.size[0] - 1.0 or
                    env.agent_pos[1] < 1.0 or env.agent_pos[1] > env.size[1] - 1.0):
                    angular_vel = 0.5
            elif i < 100:  # Turn right
                forward_vel = 0.2
                angular_vel = 0.3
            elif i < 150:  # Move diagonally up-right
                forward_vel = 0.3
                angular_vel = 0.0
                if (env.agent_pos[0] < 1.0 or env.agent_pos[0] > env.size[0] - 1.0 or
                    env.agent_pos[1] < 1.0 or env.agent_pos[1] > env.size[1] - 1.0):
                    angular_vel = -0.5
            else:  # Turn left
                forward_vel = 0.2
                angular_vel = -0.3
            
            true_state, _, collision = env.step(np.array([forward_vel, angular_vel]))

            # Optional: handle collisions gracefully if your world still has walls
            if collision:
                # Simple bounce logic: reverse and turn away
                forward_vel = -0.2
                angular_vel = 0.8
                true_state, _, _ = env.step(np.array([forward_vel, angular_vel]))

            # Take single noisy range measurement to the station
            z = env.noisy_range_to_station(true_state[:2])

            # EKF prediction and update using single range
            ekf.predict(forward_vel, angular_vel, env.dt)
            z_pred = ekf._measurement_model(ekf.mu)
            innovation = float(z - z_pred[0])
            measurement_innovations.append(abs(innovation))
            ekf.update(np.array([z]))

            # Occasionally print debug info
            if i % 50 == 0:
                print(f"\nStep {i} Debug Info:")
                print(f"True state: {true_state}")
                print(f"EKF state: {ekf.mu}")
                print(f"Measured range: {z:.3f}, Predicted: {z_pred[0]:.3f}, Innovation: {innovation:.3f}")

                        
            # Calculate errors
            pos_error = np.linalg.norm(true_state[:2] - ekf.mu[:2])
            heading_error = np.abs((true_state[2] - ekf.mu[2] + np.pi) % (2*np.pi) - np.pi)
            position_errors.append(pos_error)
            heading_errors.append(heading_error)
            
            
            # Update plots with current state
            vis.update_plots(true_state)
            
            # Update metrics in title
            window_size = 10
            avg_pos_error = np.mean(position_errors[-window_size:]) if len(position_errors) > window_size else pos_error
            avg_heading_error = np.mean(heading_errors[-window_size:]) if len(heading_errors) > window_size else heading_error
            avg_innovation = np.mean(measurement_innovations[-window_size:]) if len(measurement_innovations) > window_size else measurement_innovations[-1]
            
            vis.ax.set_title(
                f'Time: {t:.1f}s | Range station: {env.station}\n'
                f'Position Error: {pos_error:.2f} m (Avg: {avg_pos_error:.2f} m)\n'
                f'Heading Error: {np.degrees(heading_error):.1f}° (Avg: {np.degrees(avg_heading_error):.1f}°)\n'
                f'Innovation |z - ẑ|: {abs(innovation):.3f} m (Avg: {avg_innovation:.3f} m)'
            )

            
            # Use a single draw and pause call per frame
            if i % 2 == 0:  # Update display every other frame for smoother performance
                plt.draw()
                plt.pause(0.001)  # Shorter pause for better responsiveness
            
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    finally:
        # Plot error analysis
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(times, position_errors)
        plt.title('Position Error Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (m)')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(times, np.degrees(heading_errors))
        plt.title('Heading Error Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (degrees)')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(times, measurement_innovations)
        plt.title('Average Measurement Innovation')
        plt.xlabel('Time (s)')
        plt.ylabel('Innovation (m)')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(times, velocity_scales)
        plt.title('Velocity Scaling Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Scale Factor')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    test_ekf_visualization() 