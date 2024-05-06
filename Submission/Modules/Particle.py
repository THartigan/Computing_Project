import numpy as np

class Particle():
    """Initialises a particle
    properties
    ----------
    position: [float]: the position of the particle, either in 2D or 3D
    property: float: the property of that particle, e.g. mass or charge
    """
    def __init__(self, position, property):
        self.position = position
        self.complex_position = self.position[0] + 1J * self.position[1]
        self.property = property
        self.acceleration = np.array([0.,0.,0.])
        self.velocity = np.array([0.,0.,0.])
        self.total_potential = 0