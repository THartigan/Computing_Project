import numpy as np
import matplotlib.pyplot as plt

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
            #print(len(self.particles))
            other_particles = list(self.particles)
            other_particles.remove(particle)
            #print(len(other_particles))
            for other_particle in other_particles:
                particle.total_potential += -other_particle.property * np.real(np.log(particle.complex_position - other_particle.complex_position))

    def plot_potential(self):
        xs = []
        ys = []
        potentials = []
        for particle in self.particles:
            position = particle.position
            xs.append(position[0])
            ys.append(position[1])
            potentials.append(particle.total_potential)
        fig, ax = plt.subplots()
        ax.tricontour(xs, ys, potentials, levels=4, colors='k')
        cntr = ax.tricontourf(xs, ys, potentials, levels=1000, cmap="RdBu_r")
        fig.colorbar(cntr, ax=ax)
        print(min(potentials))

class Analytic():
    def __init__(self, box_width, particles):
        self.box_width = box_width
        self.particles = particles
        self.sim_box = SimBox(self.box_width)
        
    def run(self):
        for particle in self.particles:
            self.sim_box.load_particle(particle)
        self.sim_box.calc_potentials()
    
    def plot_potential(self):
        self.sim_box.plot_potential()