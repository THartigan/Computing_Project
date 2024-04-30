import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def plot_difference(particles, other_particles, z_range = [None,None], z_levels = 1000, x_range = [None, None], y_range = [None, None], cmap = 'RdBu_r'):
    xs = []
    ys = []
    potential_differences = []
    is_bad = []
    for i, particle in enumerate(particles):
        position = particle.position
        xs.append(position[0])
        ys.append(position[1])
        
        potential_differences.append(particle.total_potential - other_particles[i].total_potential)
    
    plot(xs, ys, potential_differences, z_range, z_levels, x_range, y_range, cmap)
    # triang = tri.Triangulation(xs, ys)
    # #is_bad = np.array(is_bad)
    # #mask = np.all(np.where(is_bad[triang.triangles], True, False), axis=1)
    # #triang.set_mask(mask)
    # fig, ax = plt.subplots()
    # #levels = np.linspace(-10,0,1000)
    # print("potential differences", potential_differences)
    # foo = np.array([xs, ys]).transpose()
    # print(np.array(potential_differences).transpose(), foo)
    # potential_differences = np.nan_to_num(potential_differences)
    # print(max(potential_differences))
    # print(min(potential_differences))
    # #potential_differences[abs(potential_differences) > 1E5] = 0
    # match plot_range:
    #     case [None, None]:
    #         levels = 1000
    #     case x if x[0]==None:
    #         levels = np.linspace(min(potential_differences), x[1])
    #     case x if x[1]==None:
    #         levels = np.linspace(x[0], max(potential_differences))
    #     case _:
    #         levels = np.linspace(plot_range[0], plot_range[1], 1000)
    # # if plot_range[None] != 0 and plot_range[1] != None:
        
    # # else:
    # #     levels = 1000
    # cntr = ax.tricontourf(triang, potential_differences, levels=levels, cmap='RdBu_r', )
    # #ax.tricontour(triang, potential_differences, levels=4, colors='k')
    # fig.colorbar(cntr, ax=ax)
    # non_zero_pds = potential_differences[potential_differences != 0]
    # #print(np.min(non_zero_pds))

def plot_potentials(particles, z_range = [None, None], z_levels = 1000, x_range = [None, None], y_range = [None, None], cmap = 'RdBu_r'):
    xs = []
    ys = []
    potentials = []
    for i, particle in enumerate(particles):
        position = particle.position
        xs.append(position[0])
        ys.append(position[1])
        potentials.append(particle.total_potential)
    
    plot(xs, ys, potentials, z_range, z_levels, x_range, y_range, cmap)
    

def plot(xs, ys, potentials, z_range, z_levels, x_range, y_range, cmap):
    keep_indices = np.invert(np.isinf(potentials) + np.isnan(potentials))
    potentials = np.array(potentials)[keep_indices]
    xs = np.array(xs)[keep_indices]
    ys = np.array(ys)[keep_indices]

    match z_range:
        case [None, None]:
            levels = z_levels
        case x if x[0]==None:
            levels = np.linspace(min(potentials), x[1], z_levels)
        case x if x[1]==None:
            levels = np.linspace(x[0], max(potentials), z_levels)
        case _:
            levels = np.linspace(z_range[0], z_range[1], z_levels)
    
    fig, ax = plt.subplots()
    triang = tri.Triangulation(xs, ys)
    cntr = ax.tricontourf(triang, potentials, levels=levels, cmap=cmap)
    #ax.tricontour(triang, potential_differences, levels=4, colors='k')
    fig.colorbar(cntr, ax=ax)
    
    if x_range[0] != None:
        ax.set_xlim(left=x_range[0])
    if x_range[1] != None:
        ax.set_xlim(right=x_range[1])
    if y_range[0] != None:
        ax.set_ylim(bottom=y_range[0])
    if y_range[1] != None:
        ax.set_ylim(top=y_range[1])
    
    print("Mean: ", np.mean(potentials))
    print("Median: ", np.median(potentials))
    print("Min: ", np.min(potentials))
    print("Max: ", np.max(potentials))

def plot_log_relative_errors(particles, reference_particles, z_range = [None, None], z_levels = 1000, x_range = [None, None], y_range = [None, None], cmap = 'jet'):
    xs = []
    ys = []
    potential_differences = []
    reference_potentials = []
    for i, particle in enumerate(particles):
        position = particle.position
        xs.append(position[0])
        ys.append(position[1])
        
        potential_differences.append(particle.total_potential - reference_particles[i].total_potential)
        reference_potentials.append(reference_particles[i].total_potential)
    log_rel_errors = np.log10(np.abs(np.array(potential_differences) / np.array(reference_potentials)))
    plot(xs, ys, log_rel_errors, z_range, z_levels, x_range, y_range, cmap)
    # triang = tri.Triangulation(xs, ys)
    # #is_bad = np.array(is_bad)
    # #mask = np.all(np.where(is_bad[triang.triangles], True, False), axis=1)
    # #triang.set_mask(mask)
    # fig, ax = plt.subplots()
    # #levels = np.linspace(-10,0,1000)
    # print(potential_differences)
    # relative_differences = np.abs(np.array(potential_differences) / np.array(reference_potentials))
    # log_potential_differences = np.log10(relative_differences)
    # log_potential_differences = np.nan_to_num(log_potential_differences)
    # print(max(log_potential_differences))
    # print(min(log_potential_differences))
    # #potential_differences[abs(potential_differences) > 1E5] = 0
    # match plot_range:
    #     case [None, None]:
    #         levels = 1000
    #     case x if x[0]==None:
    #         levels = np.linspace(min(log_potential_differences), x[1])
    #     case x if x[1]==None:
    #         levels = np.linspace(x[0], max(log_potential_differences))
    #     case _:
    #         levels = np.linspace(plot_range[0], plot_range[1], 1000)
    # # if plot_range[None] != 0 and plot_range[1] != None:
        
    # # else:
    # #     levels = 1000
    # cntr = ax.tricontourf(triang, log_potential_differences, levels=levels, cmap='jet', )
    # #ax.tricontour(triang, potential_differences, levels=4, colors='k')
    # fig.colorbar(cntr, ax=ax)
    # non_zero_pds = log_potential_differences[log_potential_differences != 0]
    # print(np.min(non_zero_pds))