import numpy as np
from modules.Particle import Particle

class BH():
    def __init__(self, box_size, initial_particles, theta=1):
        self.box_size = box_size
        self.particles = initial_particles
        self.theta = theta
        self.system = System(initial_particles, box_size, theta)
    
    def run(self):
        self.system.generate_tree_and_place_particles()
        self.system.calculate_box_masses()
        self.system.calculate_particle_accelerations()


class System():
    def __init__(self, initial_particles, box_size, theta) -> None:
        self.all_boxes = []
        self.box_tracker = [[]]
        self.G = 1
        self.theta = theta
        self.particles = initial_particles
        self.box_size = box_size

    def generate_tree_and_place_particles(self):
        self.root_cell = Box(self, None, np.array([0.5,0.5,0.5])*self.box_size, self.box_size, particles=[])
        for i, particle in enumerate(self.particles):
            self.root_cell.load_particle(particle)
    
    def calculate_box_masses(self):
        for tracker_slice in reversed(self.box_tracker):
            for box in tracker_slice:
                if box.unpartitioned:
                    if box.particles == []:
                        box.mass = 0
                        box.com_position = box.centre
                    else:
                        box.mass = box.particles[0].property
                        box.com_position = box.particles[0].position
                else:
                    total_mass = 0
                    position_unnormalised = np.array([0.,0.,0.])
                    for child_box in box.children:
                        total_mass += child_box.mass
                        position_unnormalised += child_box.mass * child_box.com_position
                    box.com_position = position_unnormalised / total_mass
                    box.mass = total_mass
        
    def calculate_particle_accelerations(self):
        for particle in self.root_cell.particles:
            self.root_cell.find_acceleration_for(particle, self.theta)
    
    def distance(self, box, particle: Particle):
        if np.isnan(np.linalg.norm(particle.position - box.com_position)):
            print(particle)
        return np.linalg.norm(particle.position - box.com_position)

    def newton_acceleration(self, box, particle: Particle):
        direction = (box.com_position - particle.position)/np.linalg.norm(box.com_position - particle.position)
        return self.G * box.mass / (self.distance(box, particle) ** 2) * direction

class Box():
    def __init__(self, system: System, parent, centre, size, particles = [], depth = 0) -> None:
        """Initialises a new Box object
        parent: Box - The parent of the box, ie which box was subdivided to construct this one
        centre: [Float] - The coordinate position of the centre of the box
        size: Float - The distance between parallel sides of the box
        particles: [Particle] A list of the particles contained within that box"""
        self.parent = parent
        self.system = system
        self.children = []
        self.centre = centre
        self.size = size
        self.particles = particles
        self.unpartitioned = True
        self.depth = depth
        system.all_boxes.append(self)
        if depth+1 > len(system.box_tracker):
            system.box_tracker.append([])
        system.box_tracker[depth].append(self)

    
    def load_particle(self, particle):
        self.particles.append(particle)
        if len(self.particles) > 1:
            if self.children == []:
                self.division_cell()
            else:
                self.place_particle(particle)
                
    def division_cell(self):
        self.unpartitioned = False
        new_size = self.size / 2

        new_centres = []
        centre_alterations = [new_size / 2, -new_size / 2]
        for x_centre_alteration in centre_alterations:
            for y_centre_alteration in centre_alterations:
                for z_centre_alteration in centre_alterations:
                    new_centres.append(np.array([self.centre[0] + x_centre_alteration, self.centre[1] + y_centre_alteration, self.centre[2] + z_centre_alteration]))

        for new_centre in new_centres:
            new_box = Box(self.system, self, new_centre, new_size, particles=[], depth=self.depth+1)
            self.children.append(new_box)

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
        self.children[placement_box].load_particle(particle)
    
    def find_acceleration_for(self, particle, theta):
        # If this box has a single particle, which is not the subject particle itself, then add the force form this box
        if len(self.particles) == 1 and self.particles != []:
            if self.particles[0] != particle:
                delta_acc = self.system.newton_acceleration(self, particle)
                particle.acceleration += delta_acc
        # If this box contains multiple particles and is far away, then apprxomiate this group at its COM
        elif self.size / self.system.distance(self, particle) < theta:
            delta_acc = self.system.newton_acceleration(self, particle)
            particle.acceleration += delta_acc
        # Else, the box contains many nearby particles and higher precision is required, so analyse children
        else:
            for child in self.children:
                child.find_acceleration_for(particle, theta)