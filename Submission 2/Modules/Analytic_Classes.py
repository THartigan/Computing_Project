import numpy as np
import matplotlib.pyplot as plt

class SingleParticle():
    def __init__(self, position, property, box_size) -> None:
        self.position = position
        self.property = property
        self.box_size = box_size
        self.complex_position = self.position[0] + 1j * self.position[1]
    
    def evaluate_particle_potentials(self, particles):
        for particle in particles:
            particle.total_potential = -(self.property * np.log(np.complex256(abs(particle.complex_position - self.complex_position))))
    
    def evaluate_particle_acceleration(self, particles):
        for particle in particles:
            delta_positions = self.position - particle.position
            direction = (delta_positions)/np.linalg.norm(delta_positions)
            particle.acceleration = self.property / (np.linalg.norm(delta_positions) ** 2) * direction

    def evaluate_position_potential(self, x, y):
        return -self.property * np.log(np.clongdouble(abs(x + 1j*y - self.complex_position)))
    
    def plot_potential(self):
        n = 100
        xs = np.linspace(0, self.box_size, n)
        ys = np.linspace(0, self.box_size, n)
        plot_xs = []
        plot_ys = []
        potentials = []
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                plot_xs.append(x)
                plot_ys.append(y)
                potentials.append(self.evaluate_position_potential(x, y))
     
        fig, ax = plt.subplots()
        cntr = ax.tricontourf(plot_xs, plot_ys, potentials, levels=1000, cmap="RdBu_r")
        cbar = fig.colorbar(cntr, ax=ax)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("Potential", rotation=270)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
