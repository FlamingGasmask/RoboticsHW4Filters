# Assignment 3: Particle Filter

This assignment focuses on implementing a Particle Filter for robot localization in a multimodal world with complex dynamics and non-Gaussian noise.

## Learning Objectives

- Understand particle-based state estimation
- Implement importance sampling and resampling
- Handle multimodal distributions
- Work with non-Gaussian noise models

## Installation
Navigate to the assignment3_particle_filter directory in terminal, then run:
```bash
pip install -e .
```

## Implementation Tasks

You need to implement several methods in `particle_filter.py`:

1. `predict(action)`: Implement the particle prediction step
   - Propagate each particle through motion model
   - Add motion noise to particles
   - Handle different actions (forward, turn)
   - Account for motion uncertainty

2. `update(readings)`: Implement the measurement update step
   - Calculate particle weights based on sensor readings
   - Normalize weights
   - Handle range-bearing measurements
   - Account for measurement noise

3. `resample()`: Implement particle resampling
   - Use low variance resampling algorithm
   - Maintain particle diversity
   - Handle edge cases (degenerate particles)
   - Reset weights after resampling

4. `estimate_state()`: Implement state estimation
   - Calculate weighted mean of particles
   - Handle circular quantities (angles)
   - Return best state estimate

## Visualization

To see a visualization of the state estimation for a randomly moving object, run the following command from the assignment3_particle_filter directory after implementing the functions:
```bash
python -m filtering_exercises_particle_filter.assignment3_particle_filter.tests.test_particle_vis
```