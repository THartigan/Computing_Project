import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.tri as tri
# coeffs = [ 1.00000000e+02, -1.75786746e+04, -1.54504900e+06, -1.81066091e+08,
#  -2.38717641e+10, -3.35707179e+12, -4.91773937e+14, -7.40977201e+16,
#  -1.13972224e+19, -1.78087168e+21, -2.81748274e+23, -4.50251021e+25,
#  -7.25524816e+27, -1.17727058e+30, -1.92166524e+32]
# n_particles = 1100
# box_size = 1000
# potentials = []
# xs = []
# ys = []
# point = 625
# for x in np.linspace(0, box_size, int(np.sqrt(n_particles)), False):
#     for y in np.linspace(0, box_size, int(np.sqrt(n_particles)), False):
#         x_0 = point
#         y_0 = point
#         z = np.sqrt((x-x_0)**2 + (y-y_0)**2)
#         if z>250:#(box_size / int(np.sqrt(n_particles)))*1:
#             xs.append(x)
#             ys.append(y)
#             potential = -coeffs[0]*np.log(z)
#             local_coeffs = copy.deepcopy(coeffs)
#             local_coeffs.remove(local_coeffs[0])
#             #print(coeffs)
#             #print(local_coeffs)
#             for i, coeff in enumerate(local_coeffs):
#                 potential += coeff / (z**(i+1))
#             potentials.append(potential)
        
# xs.append(point)
# ys.append(point)
# potentials.append(0)


# fig, ax = plt.subplots()
# ax.tricontour(xs, ys, potentials, levels=4, colors='k')
# cntr = ax.tricontourf(xs, ys, potentials, levels=1000, cmap="RdBu_r")
# ax.set_xlim([0,200])
# ax.set_ylim([0,200])
# fig.colorbar(cntr, ax=ax)
# print(min(potentials))
#plt.show()
class PointTesting():
    def __init__(self, particles, meshbox, r) -> None:
        self.meshbox = meshbox
        self.particles = particles
        self.coefficients = meshbox.mpe_coefficients
        self.r = r
        self.point = meshbox.complex_centre
        self.potentials = []

    def test_mpe(self):
        for particle in self.particles:
            z = abs(particle.complex_position - self.point)
            if z > self.r:
                particle.total_potential = -self.coefficients[0]*np.log(z)
                local_coeffs = list(copy.deepcopy(self.coefficients))
                local_coeffs.remove(local_coeffs[0])
                #print(coeffs)
                #print(local_coeffs)
                for i, coeff in enumerate(local_coeffs):
                    particle.total_potential += coeff / (z**(i+1))
            self.potentials.append(particle.total_potential)
    
    def plot(self):
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
    
    def plot_difference(self, particles, other_particles):
        xs = []
        ys = []
        potential_differences = []
        is_bad = []
        for i, particle in enumerate(particles):
            position = particle.position
            z = abs(particle.complex_position - self.meshbox.complex_centre)
            xs.append(position[0])
            ys.append(position[1])
            if z > self.r:
                potential_differences.append(particle.total_potential - other_particles[i].total_potential)
                is_bad.append(False)
            else:
                potential_differences.append(0)
                is_bad.append(True)

        triang = tri.Triangulation(xs, ys)
        #is_bad = np.array(is_bad)
        #mask = np.all(np.where(is_bad[triang.triangles], True, False), axis=1)
        #triang.set_mask(mask)
        fig, ax = plt.subplots()
        #levels = np.linspace(-10,0,1000)
        cntr = ax.tricontourf(triang, potential_differences, levels=1000, cmap='RdBu_r')
        #ax.tricontour(triang, potential_differences, levels=4, colors='k')
        fig.colorbar(cntr, ax=ax)
        non_zero_pds = potential_differences[potential_differences != 0]
        print(np.max(non_zero_pds))
