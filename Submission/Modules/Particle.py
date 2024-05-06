import numpy as np

class Particle():
    def __init__(self, position, property):
        self.position = position
        self.complex_position = self.position[0] + 1J * self.position[1]
        self.property = property
        self.acceleration = np.array([0.,0.,0.])
        self.velocity = np.array([0.,0.,0.])
        self.total_potential = 0