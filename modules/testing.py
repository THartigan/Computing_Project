import numpy as np

class PointTesting():
    def __init__(self, particles, meshbox, r=0):
        self.meshbox = meshbox
        self.particles = particles
        self.coefficients = meshbox.mpe_coefficients
        self.r = r
        self.point = meshbox.complex_centre
        self.potentials = []

    def test_mpe(self):
        for particle in self.particles:
            z = (particle.complex_position - self.point)
            #if abs(z) > self.r:
            particle.total_potential = self.coefficients[0]*np.log(z)
            for k in range(1, len(self.coefficients)):
                particle.total_potential += self.coefficients[k] / (z ** k)

            particle.total_potential = -np.real(particle.total_potential)