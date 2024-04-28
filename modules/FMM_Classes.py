import numpy as np
import math
from typing import List
import matplotlib.pyplot as plt

class Particle():
    def __init__(self, position, property):
        self.position = position
        self.complex_position = self.position[0] + 1J * self.position[1]
        self.property = property
        self.acceleration = np.array([0.,0.,0.])
        self.velocity = np.array([0.,0.,0.])

class Mesh():
    def __init__(self, width, n_levels, expansion_order):
        self.width = width
        self.n_levels = n_levels
        self.expansion_order = expansion_order
        # A 3D array indexed [level][i][j]
        self.meshboxes = []
        self.initialise_mesh()
    
    def initialise_mesh(self):
        # Index main levels from 0, ie level 0 is only one box
        for level in range(0, self.n_levels+1):
            self.meshboxes.append([])
            for i in range(0, 2**level):
                self.meshboxes[level].append([])
                for j in range(0, 2**level):
                    self.add_meshbox(level, [i,j])
            for i in range(0, 2**level):
                for j in range(0, 2**level):
                    self.meshboxes[level][i][j].allocate_neighbours()
            for i in range(0, 2**level):
                for j in range(0, 2**level):
                    self.meshboxes[level][i][j].calc_i_list()
            


    def add_meshbox(self, level, level_coords):
        if level-1 > -1:
            new_meshbox_parent = self.meshboxes[level-1][int(level_coords[0]/2)][int(level_coords[1]/2)]
            new_meshbox = MeshBox(self, new_meshbox_parent, level, level_coords)
            new_meshbox_parent.children.append(new_meshbox)
        else:
            new_meshbox = MeshBox(self, None, level, level_coords)
        
        self.meshboxes[level][level_coords[0]].append(new_meshbox)
    
    def add_particle(self, particle: Particle):
        fine_level_coords = np.int64(np.floor(particle.position/self.width * 2**self.n_levels))
        self.meshboxes[self.n_levels][fine_level_coords[0]][fine_level_coords[1]].add_particle(particle)
    
    def calc_fine_mpes(self):
        for meshbox in list(np.concatenate(self.meshboxes[self.n_levels]).flat):
            meshbox.calc_fine_mpe()
    
    def calc_coarse_mpes(self):
        for level in reversed(range(0, self.n_levels)):
            print("coarse level: ", level)
            for meshbox in list(np.concatenate(self.meshboxes[level]).flat):
                meshbox.calc_coarse_mpe()
    
    def calc_local_expansions(self):
        for level in range(1, self.n_levels+1): # From 1...n-1
            print("local expansion level: ", level)
            for meshbox in list(np.concatenate(self.meshboxes[level]).flat):
                meshbox.calc_local_expansion()
        for level in range(1, self.n_levels): # From 1...n-1
            for meshbox in list(np.concatenate(self.meshboxes[level]).flat):
                meshbox.translate_le_to_children()

    def calc_le_particle_potentials(self):
        for meshbox in list(np.concatenate(self.meshboxes[self.n_levels]).flat):
            meshbox.evaluate_particle_les()
    
    def calc_neighbour_particle_potentials(self):
        for meshbox in list(np.concatenate(self.meshboxes[self.n_levels]).flat):
            meshbox.evaluate_neighbour_potentials()
    
    def plot_potential(self, plot_range = [0,0]):
        xs = []
        ys = []
        potentials = []
        for particle in self.meshboxes[0][0][0].particles:
            position = particle.position
            if particle.total_potential != 0:
                xs.append(position[0])
                ys.append(position[1])
                potentials.append(particle.total_potential)
        fig, ax = plt.subplots()

        potential_differences = np.nan_to_num(potential_differences)
        print(max(potential_differences))
        print(min(potential_differences))
        #potential_differences[abs(potential_differences) > 1E5] = 0
        if plot_range[0] != 0 or plot_range[1] != 0:
            levels = np.linspace(plot_range[0], plot_range[1], 1000)
        else:
            levels = 1000
        
        ax.tricontour(xs, ys, potentials, levels=[0], colors='k')
        cntr = ax.tricontourf(xs, ys, potentials, levels=1000, cmap="RdBu_r")
        # ax.set_xlim([0,200])
        # ax.set_ylim([0,200])
        fig.colorbar(cntr, ax=ax)
        print(min(potentials))
            
            

class MeshBox():
    def __init__(self, mesh: Mesh, parent, level, level_coords):
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
        self.psi_tilda_coeffs = np.zeros(self.mesh.expansion_order+1)
    
    def allocate_neighbours(self):
        neighbour_list = []
        for i in range(-1,2):
            for j in range(-1,2):
                if self.level_coords[0]+i > -1 and self.level_coords[1]+j > -1 and self.level_coords[0]+i < 2**self.level and self.level_coords[1]+j < 2**self.level:
                    neighbour_list.append(self.mesh.meshboxes[self.level][self.level_coords[0]+i][self.level_coords[1]+j])
        neighbour_list.remove(self)
        self.neighbours = neighbour_list
    
    def calc_i_list(self):
        i_list = [] # Interaction list
        if self.parent != None:
            for parent_neighbour in self.parent.neighbours:
                i_list += parent_neighbour.children
            for own_neighbour in self.neighbours:
                if own_neighbour in i_list:
                    i_list.remove(own_neighbour)
            self.i_list = i_list
    
    def add_particle(self, particle: Particle):
        self.particles.append(particle)
        if self.parent != None:
            self.parent.add_particle(particle)
    
    def calculate_total_property(self):
        self.property_total = sum([particle.property for particle in self.particles])
        return self.property_total
    
    def calc_fine_mpe(self): # 100% Working
        coefficients = [self.calculate_total_property()] ## ADDED MINUS HERE
        if self.mesh.expansion_order < 0:
            raise Exception("For multipole expansion, p must be greater than 0")
        for exponent in range(1, self.mesh.expansion_order+1):
            coefficient = 0
            for particle in self.particles:
                coefficient += (-particle.property * abs(particle.complex_position - self.complex_centre) ** exponent) / exponent
                #print((particle.complex_position - self.complex_centre))
                #print(f'particle {particle.complex_position}')
                #print(f'box {self.complex_centre}')
            coefficients.append(coefficient)
        # Includes a_0 (Q) up to a_n
        self.mpe_coefficients = coefficients
        if self.level == self.mesh.n_levels and coefficients[0] != 0:
            print("fine", self.level, self.level_coords, self.centre)
            print(coefficients)
            print(self)
    
    def calc_coarse_mpe(self):
        shift_coeffs = np.zeros(self.mesh.expansion_order+1)
        for child_meshbox in self.children:
            child_coeffs = child_meshbox.mpe_coefficients
            # Initialise the array and include the (unchanged) a_0
            child_shift_coeffs = [child_coeffs[0]]
            # Calculates the remaining b_l
            #print("coarse", self.level, self.level_coords, self.centre)
            for shift_exponent in range(1, self.mesh.expansion_order + 1):
                child_shift_coeff = 0
                for k in range(1, shift_exponent+1):
                    z_0 = abs(child_meshbox.complex_centre - self.complex_centre)
                    child_shift_coeff += child_coeffs[k] * (z_0) ** (shift_exponent-k) * math.comb(shift_exponent-1, k-1)
                #print(child_shift_coeff)
                child_shift_coeff += (-child_coeffs[0] * (z_0 ** shift_exponent)) / shift_exponent
                #print(child_shift_coeff)
                child_shift_coeffs.append(child_shift_coeff)
            #print(child_shift_coeffs)
            
            #print(len(shift_coeffs))
            #print(len(child_shift_coeffs))
            shift_coeffs += child_shift_coeffs
        self.mpe_coefficients = shift_coeffs
        if self.mpe_coefficients[0] != 0:
            print("coarse", self.level, self.level_coords, self.centre)
            print(self.mpe_coefficients)
            print(self)
        
        #print(shift_coeffs)
        #print(shift_coeffs)
        #print(shift_coeffs)

    def calc_local_expansion(self):
        psi_coeffs = np.zeros(self.mesh.expansion_order+1)
        p = self.mesh.expansion_order
        # Evaluate psi for the only box at level 0 for periodic boundary conditions
        # if self.level == 0:
        #     for m in range(1,p+1):
        #         for k in range(1, p+1):
        #             psi_coeffs += 1/
        psi_coeffs = np.zeros(self.mesh.expansion_order+1)
        p = self.mesh.expansion_order
        
        for i_box in self.i_list:
            z_0 = abs(i_box.complex_centre - self.complex_centre)

            # Calculating b_0
            for k in range(1, p +1):
                psi_coeffs[0] += i_box.mpe_coefficients[k] / (z_0 ** k) * ((-1)**k)
            psi_coeffs[0] += i_box.mpe_coefficients[0] * np.log(z_0)

            # Calculating b_l
            for l in range(1, p+1):
                for k in range(1, p+1):
                    psi_coeffs[l] += 1/(z_0**l) * i_box.mpe_coefficients[k] / (z_0**k) * math.comb(l+k-1, k-1) * ((-1)**k)
                psi_coeffs[l] += -i_box.mpe_coefficients[0] / (l*(z_0**l))

        self.psi_coeffs = psi_coeffs
        print("local expansion psi", self.level, self.level_coords, psi_coeffs)
        print("local expansion interaction boxes", self.i_list)

        # Adding psi to psi_tilda, unless at level 1 in which case, this is the starting expansion
        if self.level == 1:
            self.psi_tilda_coeffs = psi_coeffs
        else:
            self.psi_tilda_coeffs += psi_coeffs

        # if self.level == 1:
        #     psi_tilda = np.zeros(self.mesh.expansion_order+1) # Local expansion coefficients
        # else:
        #     psi_tilda = self.total_le_coeffs 
        


        # if self.level == 1:
        #     psi_tilda = np.zeros(self.mesh.expansion_order+1) # Local expansion coefficients
        # else:
        #     psi_tilda = self.total_le_coeffs
        #     #print("Starting with old", total_le_coeffs)

        # psi_meshbox_level = np.zeros(self.mesh.expansion_order+1)
        # for i_box in self.i_list:
        #     z_0 = abs(i_box.complex_centre - self.complex_centre)
        #     # Calculating b_0
        #     b_0 = 0
        #     for k in range(1, self.mesh.expansion_order+1):
        #         b_0 += i_box.mpe_coefficients[k]/(z_0**k) * (-1)**k
        #     b_0 += i_box.mpe_coefficients[0] * (np.log(z_0))
        #     psi_meshbox_level[0] += b_0 ## TESTING MEASURE
        #     for l in range(1, self.mesh.expansion_order+1):
        #         b_l = 0
        #         for k in range(1,self.mesh.expansion_order+1):
        #             #print(l, k)
        #             b_l += 1/(z_0**l) * i_box.mpe_coefficients[k] / (z_0**k) * math.comb(l+k-1, k-1) * ((-1)**k)
        #         b_l -= i_box.mpe_coefficients[0]/(l*(z_0**l))
        #         #if l ==1: 
        #         psi_meshbox_level[l] += b_l ##TESTING MEASURE
        # self.total_le_coeffs = psi_tilda + psi_meshbox_level # This being too small would explain lack of secondary circles
        # self.level_le_coeffs = psi_meshbox_level


        #print("local", self.level, self.level_coords, self.centre)
        #print(total_le_coeffs)
        #print(total_le_coeffs)
        #print(self.level)
        #print(total_le_coeffs)
    
    def translate_le_to_children(self):
        p = self.mesh.expansion_order
        print("parent", self.level, self.level_coords, self.psi_tilda_coeffs)
        for child_box in self.children:
            
            z_0 = child_box.complex_centre - self.complex_centre
            child_psi_tilda_coeffs = np.zeros(self.mesh.expansion_order+1)
            for l in range(0, p+1):
                for k in range(l, p+1):
                    child_psi_tilda_coeffs[l] += self.psi_tilda_coeffs[k] * math.comb(k, l) * (abs(z_0) **(k-l))
                    # child_total_le_coeffs[l] += self.level_le_coeffs[k] * math.comb(k,l) * (abs(z_0))**(k-l)
            print("child shifted coeffs", child_box.level_coords, child_psi_tilda_coeffs)
            child_box.psi_tilda_coeffs += child_psi_tilda_coeffs
            
            #child_box.total_le_coeffs = child_total_le_coeffs
            #print(child_total_le_coeffs)

    def evaluate_particle_les(self):
        for particle in self.particles:
            particle.le_potential = self.evaluate_le(particle)
    
    def evaluate_le(self, particle: Particle):
        le_potential = 0
        z = particle.complex_position - self.complex_centre
        for l in range(0, self.mesh.expansion_order+1):
            le_potential += self.psi_tilda_coeffs[l] * abs(z)**l
        #print(le_potential)
        return -le_potential
    
    def evaluate_neighbour_potentials(self):
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
            box_particle.neighbour_potential = particle_neighbour_potential#*0 ##TESTING MEASURE
            box_particle.total_potential = box_particle.neighbour_potential + box_particle.le_potential
            box_particle.real_potential = box_particle.total_potential

class FMM():
    def __init__(self, box_size, expansion_order, particles, n_levels=0) -> None:
        self.n_particles = len(particles)
        self.particles = particles
        self.box_size = box_size
        if n_levels == 0:
            self.n_levels = int(np.ceil(np.emath.logn(4, self.n_particles)))
        else:
            self.n_levels = n_levels
        self.p = expansion_order #int(np.ceil(np.log2(precision)))
        self.mesh = None

    def run(self):
        self.mesh = Mesh(self.box_size, self.n_levels, self.p)
        for particle in self.particles:
            self.mesh.add_particle(particle)

        print(1)
        self.mesh.calc_fine_mpes() # Step 1
        print(2)
        self.mesh.calc_coarse_mpes() # Step 2
        print(3)
        self.mesh.calc_local_expansions() # Step 3 and 4
        print(4)
        self.mesh.calc_le_particle_potentials() # Step 5
        print(5)
        self.mesh.calc_neighbour_particle_potentials() # Step 6 and 7
    
    def plot_potential(self):
        self.mesh.plot_potential()