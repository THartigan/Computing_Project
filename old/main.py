import numpy as np
import matplotlib.pyplot as plt
from modules.Particle import Particle
from modules.Direct_Classes import Analytic
from modules.FMM_Classes import FMM
import copy
from modules.testing import PointTesting

# Create particle distribution to be used in all simulations
#np.random.seed(100)
n_particles = 5000
max_property = 0 # property could be mass or charge 
box_size = 1000

# Generate initial particles
#initial_positions = np.random.beta(2,2,(n_particles,2)) * box_size
initial_positions=[]
for x in np.linspace(0, box_size, int(np.sqrt(n_particles)), False):
    for y in np.linspace(0, box_size, int(np.sqrt(n_particles)), False):
        initial_positions.append(np.array([x, y])) 
initial_positions = np.array(initial_positions)
print(initial_positions)
#initial_positions = np.random.uniform(0,1,(n_particles,2)) * box_size

initial_particles = []
for initial_position in initial_positions:
    initial_particles.append(Particle(initial_position, np.random.uniform(0,max_property)))

point = 515.4
initial_particles.append(Particle(np.array([point,point]), 100))

# Analytic Simulation
analytic_particles = copy.deepcopy(initial_particles)
analytic_simulation = Analytic(box_size, analytic_particles)
analytic_simulation.run()
analytic_simulation.plot_potential()
#plt.show()

# FMM Simulation
precision = 1000
fmm_particles = copy.deepcopy(initial_particles)
fmm_simulation = FMM(box_size, precision, fmm_particles)
fmm_simulation.run()
fmm_simulation.plot_potential()

#initial_particles.append(Particle([box_size/2, box_size/2], 100))
#n_particles += 1
test_meshbox = fmm_simulation.mesh.meshboxes[5][16][16]
r = (box_size / int(np.sqrt(n_particles)))*3*np.sqrt(2)*1.5
point_test = PointTesting(initial_particles, test_meshbox, r)
point_test.test()
point_test.plot()
point_test.plot_difference(analytic_simulation.particles)


# analytic_particles = analytic_simulation.particles
# test_particles = point_test.particles
# for i, analytic_particle in enumerate(analytic_particles):
#     if abs(analytic_particle.complex_position - test_meshbox.complex_centre) > r:
#         analytic_particle.total_potential -= test_particles[i].total_potential
#     else:
#         analytic_particle.total_potential = np.infty
# analytic_simulation.plot_potential()

plt.show()
