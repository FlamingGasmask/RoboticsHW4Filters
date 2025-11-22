# Assignment 1: Discrete Bayes Filter
This assignment focuses on implementing a Discrete Bayes Filter for robot localiation in a discrete gridworld with random robot movements and basic range measurements

## Learning Objectives

- build conceptual understanding of recurcive probabilistic estimation
- gain hands-on experience implementing and visualizing the update process

## Installation
Navigate to the assignment1_bayes directory in terminal, then run:
```bash
pip install -e .
```

## Implementation Tasks

You need to implement two methods in `bayes_filter.py`:

1. predict
2. update

Additionally description of what must be implemented can be found in the docstrings of the functions.

## Visualization

To see a visualization of the state estimation for a randomly moving object, run the following command from the assignment1_bayes directory after implementing the functions:
```bash
python -m filtering_exercises_bayes.assignment1_bayes.tests.test_bayes_filter_vis
```

