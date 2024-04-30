import numpy as np
import matplotlib.pyplot as plt
import modules.Utility as util

class SimBox():
    def __init__(self, width) -> None:
        self.width = width
        self.particles = []
    
    def load_particle(self, particle):
        self.particles.append(particle)

    def calc_potentials(self):
        i=0
        for particle in self.particles:
            i += 1
            if i % 1000 == 0: print(i)
            other_particles = list(self.particles)
            other_particles.remove(particle)
            for other_particle in other_particles:
                particle.total_potential += -other_particle.property * np.real(np.log(particle.complex_position - other_particle.complex_position))

    # def plot_potential(self):
    #     xs = []
    #     ys = []
    #     potentials = []
    #     for particle in self.particles:
    #         position = particle.position
    #         xs.append(position[0])
    #         ys.append(position[1])
    #         potentials.append(particle.total_potential)
    #     fig, ax = plt.subplots()
    #     ax.tricontour(xs, ys, potentials, levels=4, colors='k')
    #     cntr = ax.tricontourf(xs, ys, potentials, levels=1000, cmap="RdBu_r")
    #     fig.colorbar(cntr, ax=ax)
    #     print(min(potentials))

class Direct():
    def __init__(self, box_width, particles):
        self.box_width = box_width
        self.particles = particles
        self.sim_box = SimBox(self.box_width)
        
    def run(self, plotting = True, fig = None, ax = None, z_range = [None, None], z_levels = 1000, x_range = [None, None], y_range = [None, None], cmap = 'RdBu_r'):
        for particle in self.particles:
            self.sim_box.load_particle(particle)
        self.sim_box.calc_potentials()
        return util.calc_potential_results(self.particles, plotting, fig, ax, z_range, z_levels, x_range, y_range, cmap)
    
    # def plot_potential(self):
    #     self.sim_box.plot_potential()