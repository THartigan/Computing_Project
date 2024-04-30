import numpy as np
from modules.Particle import Particle

all_boxes = []
box_tracker = [[]]
G = 1
class BH():
    def __init__(self) -> None:
        self.all_boxes = []
        self.box_tracker = []

        
class Box():
    def __init__(self, parent, centre, size, particles = [], depth = 0) -> None:
        """Initialises a new Box object
        parent: Box - The parent of the box, ie which box was subdivided to construct this one
        centre: [Float] - The coordinate position of the centre of the box
        size: Float - The distance between parallel sides of the box
        particles: [Particle] A list of the particles contained within that box"""
        self.parent = parent
        self.children = []
        self.centre = centre
        self.size = size
        self.particles = particles
        self.unpartitioned = True
        self.depth = depth
        all_boxes.append(self)
        if depth+1 > len(box_tracker):
            box_tracker.append([])
        #print(depth)
        box_tracker[depth].append(self)

    
    def load_particle(self, particle):
        self.particles.append(particle)
        #print(self)
        #print(self.particles)
        if len(self.particles) > 1:
            if self.children == []:
                self.division_cell()
            else:
                self.place_particle(particle)
                
    def division_cell(self):
        #print(f"Splitting depth {self.depth} box")
        self.unpartitioned = False
        new_size = self.size / 2

        new_centres = []
        centre_alterations = [new_size / 2, -new_size / 2]
        for x_centre_alteration in centre_alterations:
            for y_centre_alteration in centre_alterations:
                for z_centre_alteration in centre_alterations:
                    new_centres.append(np.array([self.centre[0] + x_centre_alteration, self.centre[1] + y_centre_alteration, self.centre[2] + z_centre_alteration]))

        for new_centre in new_centres:
            new_box = Box(self, new_centre, new_size, particles=[], depth=self.depth+1)
            self.children.append(new_box)
            #all_boxes.append(new_box)

        for particle in self.particles:
            self.place_particle(particle)

    def place_particle(self, particle):
        particle_map = "".join(np.where(particle.position > self.centre, "1", "0"))
        particle_to_box_dict = {"111" : 0,
                                "110" : 1,
                                "101" : 2,
                                "100" : 3,
                                "011" : 4,
                                "010" : 5,
                                "001" : 6,
                                "000" : 7}
        placement_box = particle_to_box_dict[particle_map]
        #print(f"adding {particle} to box {placement_box}")
        self.children[placement_box].load_particle(particle)
    
    def find_acceleration_for(self, particle, theta):
        # If this box has a single particle, which is not the subject particle itself, then add the force form this box
        if len(self.particles) == 1 and self.particles != []:
            if self.particles[0] != particle:
                delta_acc = newton_acceleration(self, particle)
                particle.acceleration += delta_acc
        # If this box contains multiple particles and is far away, then apprxomiate this group at its COM
        elif self.size / distance(self, particle) < theta:
            delta_acc = newton_acceleration(self, particle)
            particle.acceleration += delta_acc
        # Else, the box contains many nearby particles and higher precision is required, so analyse children
        else:
            for child in self.children:
                child.find_acceleration_for(particle, theta)

# class Particle():
#     def __init__(self, position, mass) -> None:
#         self.position = position
#         self.mass = mass
#         self.acceleration = np.array([0.,0.,0.])
#         self.velocity = np.array([0.,0.,0.])

def distance(box: Box, particle: Particle):
    #print(particle.position)
    #print(box.com_position)
    return np.linalg.norm(particle.position - box.com_position)

def newton_acceleration(box: Box, particle: Particle):
    direction = (box.com_position - particle.position)/np.linalg.norm(box.com_position - particle.position)
    return G * box.mass / (distance(box, particle) ** 2) * direction


