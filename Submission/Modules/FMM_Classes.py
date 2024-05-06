import numpy as np
import math
from typing import List
import modules.Utility as util
from modules.Particle import Particle

class Mesh():
    def __init__(self, width, n_levels, expansion_order):
        """Defines a mesh object, that contains all boxes of a FMM simulation
        properties
        ----------
        width: float: the width of the system
        n_levels: Int: the number of level to which the meshbox should extend
        expansion_order: Int: the number of expansions which should occur for each MPE"""
        self.width = width
        self.n_levels = n_levels
        self.expansion_order = expansion_order
        self.meshboxes = [] # Will for a 3D array indexed [level][i][j]
        self.initialise_mesh()
    
    def initialise_mesh(self):
        """Initialises a Mesh Object"""
        # Index main levels from 0, ie level 0 is only one box
        # Generates the initial meshbox structure
        for level in range(0, self.n_levels+1):
            self.meshboxes.append([])
            for i in range(0, 2**level):
                self.meshboxes[level].append([])
                for j in range(0, 2**level):
                    self.add_meshbox(level, [i,j])

    def add_meshbox(self, level, level_coords):
        """Adds an additional meshbox at the specified level
        and with the specified level coordinates"""
        # Allocate parent and child boxes after production of the new meshbox
        if level-1 > -1:
            new_meshbox_parent = self.meshboxes[level-1][int(level_coords[0]/2)][int(level_coords[1]/2)]
            new_meshbox = MeshBox(self, new_meshbox_parent, level, level_coords)
            new_meshbox_parent.children.append(new_meshbox)
        else:
            new_meshbox = MeshBox(self, None, level, level_coords)
        
        # Keep track of the meshboxes
        self.meshboxes[level][level_coords[0]].append(new_meshbox)
    
    def add_particle(self, particle: Particle):
        """Adds a particle to the simulation"""
        fine_level_coords = np.int64(np.floor(particle.position/self.width * 2**self.n_levels))
        self.meshboxes[self.n_levels][fine_level_coords[0]][fine_level_coords[1]].add_particle(particle)
    
    def calc_fine_mpes(self):
        """Calculates the fine level multipole expansions"""
        for meshbox in list(np.concatenate(self.meshboxes[self.n_levels]).flat):
            meshbox.calc_fine_mpe()
    
    def calc_coarse_mpes(self):
        """Calculates the coarse level multipole expansions"""
        for level in reversed(range(0, self.n_levels)):
            for meshbox in list(np.concatenate(self.meshboxes[level]).flat):
                meshbox.calc_coarse_mpe()
    
    def calc_local_expansions(self):
        """Calculates the local expansions within each box"""
        for level in range(0, self.n_levels+1):
            for i in range(0, 2**level):
                    for j in range(0, 2**level):
                        # Neighbours must be allocated first
                        self.meshboxes[level][i][j].allocate_neighbours()
            for i in range(0, 2**level):
                for j in range(0, 2**level):
                    # Interaction lists must also be generated
                    self.meshboxes[level][i][j].calc_i_list()

        # Local expansions are then calculated
        for level in range(1, self.n_levels+1): # From 1...n-1
            for meshbox in list(np.concatenate(self.meshboxes[level]).flat):
                meshbox.calc_local_expansion()
            if level != self.n_levels:
                pass
                for meshbox in list(np.concatenate(self.meshboxes[level]).flat):
                    meshbox.translate_le_to_children()

    def calc_le_particle_potentials(self):
        """calculates the local expansion potentials"""
        for meshbox in list(np.concatenate(self.meshboxes[self.n_levels]).flat):
            meshbox.evaluate_particle_les()
    
    def calc_neighbour_particle_potentials(self):
        """calculates neighbour particle potential contributions"""
        for meshbox in list(np.concatenate(self.meshboxes[self.n_levels]).flat):
            meshbox.evaluate_neighbour_potentials()
            
            

class MeshBox():
    def __init__(self, mesh: Mesh, parent, level, level_coords):
        """Initialises a meshbox object
        parameters
        ----------
        mesh: Mesh: the mesh within which this meshbox exists
        parent: MeshBox: the parent meshbox
        level: the level within the mesh structure this meshbox is at
        level_coords: the coordinates of this meshbox within the level
        """
        self.mesh = mesh
        self.level = level
        self.level_coords = level_coords
        self.width = mesh.width / (2 ** level)
        self.children = []
        self.parent = parent
        self.particles = []
        self.centre = (np.array(level_coords) + np.array([0.5, 0.5])) / 2**level * mesh.width
        self.complex_centre = self.centre[0] + 1j*self.centre[1]
        self.i_list : List[MeshBox] = []
        self.neighbours = []
        self.total_le_coeffs = np.zeros(self.mesh.expansion_order+1, dtype='complex256')
    
    def allocate_neighbours(self):
        """Allocates neighbour boxes"""
        neighbour_list = []
        for i in range(-1,2):
            for j in range(-1,2):
                if self.level_coords[0]+i > -1 and self.level_coords[1]+j > -1 and self.level_coords[0]+i < 2**self.level and self.level_coords[1]+j < 2**self.level:
                    neighbour_list.append(self.mesh.meshboxes[self.level][self.level_coords[0]+i][self.level_coords[1]+j])
        # Do not count self as a neighbour
        while self in neighbour_list:
            neighbour_list.remove(self)
        self.neighbours = neighbour_list
    
    def calc_i_list(self):
        """Calcualte the interaction list of this meshbox"""
        i_list = [] # Interaction list
        if self.parent != None:
            for parent_neighbour in self.parent.neighbours:
                i_list += parent_neighbour.children
            for own_neighbour in self.neighbours:
                if own_neighbour in i_list:
                    i_list.remove(own_neighbour)
            self.i_list = i_list
    
    def add_particle(self, particle: Particle):
        """Add a particle to this meshbox"""
        self.particles.append(particle)
        if self.parent != None:
            self.parent.add_particle(particle)
    
    def calculate_total_property(self):
        """Calculate the sum of the properties of the particles within this meshbox"""
        self.property_total = sum([particle.property for particle in self.particles])
        return self.property_total
    
    def calc_fine_mpe(self):
        """Calculate the fine level multipole expansion for this box"""
        coefficients = [self.calculate_total_property()]
        if self.mesh.expansion_order < 0:
            raise Exception("For multipole expansion, p must be greater than 0")
        for exponent in range(1, self.mesh.expansion_order+1):
            coefficient = 0
            for particle in self.particles:
                coefficient += -particle.property * ((particle.complex_position - self.complex_centre) ** exponent) / exponent
            coefficients.append(coefficient)
        self.mpe_coefficients = coefficients
    
    def calc_coarse_mpe(self):
        """Calculate the coarse level multipole expansion for this box"""
        shift_coeffs = np.zeros(self.mesh.expansion_order+1, dtype='complex256')
        for child_meshbox in self.children:
            child_coeffs = child_meshbox.mpe_coefficients
            child_shift_coeffs = [child_coeffs[0]]
            # Calculates the remaining b_l
            for shift_exponent in range(1, self.mesh.expansion_order + 1):
                child_shift_coeff = 0
                for k in range(1, shift_exponent+1):
                    z_0 = (child_meshbox.complex_centre - self.complex_centre)
                    child_shift_coeff += child_coeffs[k] * (z_0) ** (shift_exponent-k) * math.comb(shift_exponent-1, k-1)
                child_shift_coeff += (-child_coeffs[0] * (z_0 ** shift_exponent)) / shift_exponent
                child_shift_coeffs.append(child_shift_coeff)
            shift_coeffs += child_shift_coeffs
        self.mpe_coefficients = shift_coeffs

    def calc_local_expansion(self):
        """Calculates the local expansion for this meshbox"""
        self.le_coeffs_from_iboxes = np.zeros(self.mesh.expansion_order+1, dtype='complex256')
        p = self.mesh.expansion_order
        
        for i_box in self.i_list:
            le_coeffs_from_i_box = np.zeros(self.mesh.expansion_order+1, dtype='complex256')
            z_0 = (i_box.complex_centre - self.complex_centre)

            # Calculating b_0
            for k in range(1, p +1):
                le_coeffs_from_i_box[0] += i_box.mpe_coefficients[k] / (z_0 ** k) * ((-1)**k)
            le_coeffs_from_i_box[0] += i_box.mpe_coefficients[0] * np.log(z_0)

            # Calculating b_l
            for l in range(1, p+1):
                for k in range(1, p+1):
                    le_coeffs_from_i_box[l] += 1/(z_0**l) * i_box.mpe_coefficients[k] / (z_0**k) * np.clongdouble(math.comb(l+k-1, k-1) * ((-1)**k))
                le_coeffs_from_i_box[l] += -i_box.mpe_coefficients[0] / (l*(z_0**l))

            self.le_coeffs_from_iboxes += le_coeffs_from_i_box
        
        # Adding psi to psi_tilda
        self.total_le_coeffs += self.le_coeffs_from_iboxes
    
    def translate_le_to_children(self):
        """Translates the local expansion of this meshbox to its children"""
        p = self.mesh.expansion_order
        for child_box in self.children:
            z_0 = child_box.complex_centre - self.complex_centre
            child_le_coeffs_from_parent = np.zeros(self.mesh.expansion_order+1, dtype='complex256')
            for l in range(0, p+1):
                for k in range(l, p+1):
                    child_le_coeffs_from_parent[l] += self.total_le_coeffs[k] * math.comb(k, l) * ((z_0) **(k-l))
            child_box.total_le_coeffs += child_le_coeffs_from_parent
            
    def evaluate_particle_les(self):
        """Evaluate the prticle local expansions within this meshbox"""
        for particle in self.particles:
            particle.le_potential = self.evaluate_le(particle)
    
    def evaluate_le(self, particle: Particle):
        """Evaluate the local expansion within this meshbox due to non-neighbour particles"""
        le_potential = 0
        z = particle.complex_position - self.complex_centre
        for l in range(0, self.mesh.expansion_order+1):
            le_potential += np.real(self.total_le_coeffs[l] * (z)**l)
        return -le_potential
    
    def evaluate_neighbour_potentials(self):
        """Evlaute the potential due to neighbour particles, and add it to the potential without them"""
        box_particles = self.particles
        interaction_particles = []
        interaction_particles += box_particles 
        for neighbour in self.neighbours:
            interaction_particles += neighbour.particles
        for box_particle in box_particles:
            particle_neighbour_potential = 0
            for neighbour_particle in interaction_particles:
                if neighbour_particle != box_particle:
                    particle_neighbour_potential += neighbour_particle.property * np.real(-np.log(box_particle.complex_position - neighbour_particle.complex_position))
            box_particle.neighbour_potential = particle_neighbour_potential
            box_particle.total_potential = box_particle.neighbour_potential + box_particle.le_potential

class FMM():
    def __init__(self, box_size, particles, n_levels=0, precision=None, p=None) -> None:
        """Initialises the FMM class, which can be used to simulate a FMM process"""
        self.n_particles = len(particles)
        self.particles = particles
        self.box_size = box_size
        if n_levels == 0:
            self.n_levels = int(np.ceil(np.emath.logn(4, self.n_particles)))
        else:
            self.n_levels = n_levels
        if p == None and precision == None:
            raise Exception("Either p or precision must be specified")
        
        if p == None: # If precision is specified
            self.p = int(np.ceil(np.log2(precision)))
            self.expected_precision = precision
        else: # If p is specified
            self.p = p
            self.expected_precision = 2 ** (-p)
        self.log_10_expected_precision = np.log10(self.expected_precision)

        self.mesh = None

    def run(self, plotting = True, fig = None, ax = None, z_range = [None, None], z_levels = 1000, x_range = [None, None], y_range = [None, None], cmap = 'RdBu_r'):
        """Runs the FMM simulation"""
        self.mesh = Mesh(self.box_size, self.n_levels, self.p)
        for particle in self.particles:
            self.mesh.add_particle(particle)

        self.mesh.calc_fine_mpes() # Step 1
        self.mesh.calc_coarse_mpes() # Step 2
        self.mesh.calc_local_expansions() # Step 3 and 4
        self.mesh.calc_le_particle_potentials() # Step 5
        self.mesh.calc_neighbour_particle_potentials() # Step 6 and 7
        return util.calc_potential_results(self.particles, plotting, fig, ax, z_range, z_levels, x_range, y_range, cmap)