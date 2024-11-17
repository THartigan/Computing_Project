import numpy as np

class PointTesting():
    def __init__(self, particles, meshbox, r=0):
        """Initialises a PointTesting object
        parameters
        ----------
        particles: [Particle]: a list of particles to be tested
        meshbox: Meshbox: the meshbox about which the multipole expansion is
        to be calcualted"""
        self.meshbox = meshbox
        self.particles = particles
        self.coefficients = meshbox.mpe_coefficients
        self.r = r
        self.point = meshbox.complex_centre
        self.potentials = []

    def test_mpe(self):
        """Set the potential for each particle in particles to that due to
        the multipole expansion about meshbox"""
        for particle in self.particles:
            z = (particle.complex_position - self.point)
            #Implementation of MPE formulae
            particle.total_potential = self.coefficients[0]*np.log(z)
            for k in range(1, len(self.coefficients)):
                particle.total_potential += self.coefficients[k] / (z ** k)

            particle.total_potential = -(np.real(particle.total_potential))