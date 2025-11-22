# Assignment 2: Extended Kalman Filter

This assignment focuses on implementing an Extended Kalman Filter (EKF) for robot localization in a continuous world with nonlinear dynamics.

## Learning Objectives

- Understand nonlinear state estimation
- Implement linearization through Jacobian matrices
- Handle continuous state spaces
- Work with Gaussian distributions and uncertainty

## Installation
Navigate to the assignment2_ekf directory in terminal, then run:

```bash
pip install -e .
```

## Implementation Tasks

You need to implement several methods in `extended_kalman_filter.py`:

1. motion_model
2. motion_jacobian
3. measurement_jacobian
4. predict
5. update


## Visualization
To see a visualization of the state estimation for a randomly moving object, run the following command from the assignment2_ekf directory after implementing the functions:
```bash
python -m filtering_exercises_ekf.assignment2_ekf.tests.test_ekf_vis
```

