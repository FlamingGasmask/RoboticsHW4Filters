from filtering_exercises.environments import MultiModalWorld
from filtering_exercises.assignment3_particle import ParticleFilter
from filtering_exercises.utils import FilterVisualizer

# Create environment and filter
env = MultiModalWorld()
pf = ParticleFilter(env, num_particles=100)

# Visualize performance
vis = FilterVisualizer(env, pf)
vis.visualize_episode() 