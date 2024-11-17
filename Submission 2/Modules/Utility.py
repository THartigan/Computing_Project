# Provides convenience function for plotting purposes as well as the results class
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.ticker as mtick

class Results():
    def __init__(self, max = None, min = None, mean = None, median = None, fig = None, ax = None):
        """
        parameters
        ----------
        max: float: the maximum value of the result
        min: float: the minimum value of the result
        mean: float: the mean value of the result
        median: float: the median value of the result
        fig: matplotlib figure object: the figure on which this result is to be plotted
        ax: matplotlib axes object: the axes object on which the result has been plotted
        """
        self.max = max
        self.min = min
        self.mean = mean
        self.median = median
        self.fig = fig
        self.ax = ax
        self.sim_time = None

def calc_potential_results(particles, plotting = True, fig = None, ax = None, z_range = [None, None], 
                           z_levels = 1000, x_range = [None, None], y_range = [None, None], cmap = 'RdBu_r', 
                           x_label = "x / Arbitary Units", y_label ="y / Arbitary Units", z_label = "Potential", 
                           title = "", axis_label = None):
    """
    Returns the result of plotting the particles
    Parameters
    ----------
    particles [Particle]: the particles to be plotted
    ploting = True: Bool: whether or not to plot the particle
    fig: matplotlib figure object: the figure on which to plot the result if provided and plotting=True
    ax: matplotlib axes object: the axes on which to plot the result if provided and plotting=True
    z_range: [float, float]: the z range to be shown on a graph
    z_levels: Int: the number of z levels to be shown on the plot
    x_range: [float, float]: the x range to be plotted
    y_ragnge: [float, float]: the y range to be plotted
    cmap: String: the matplotlib colour map to use
    x_label: String: the x axis label
    y_label: String: the y axis label
    z_label: String: the z axis label
    title: String: the title of the graph
    axis_label: String: the axis label, e.g. a) in multi-subplot graphs
    Returns
    -------
    Results object
    """
    xs = []
    ys = []
    potentials = []
    for i, particle in enumerate(particles):
        position = particle.position
        xs.append(position[0])
        ys.append(position[1])
        potentials.append(particle.total_potential)
    return analyse_2D(xs, ys, potentials, plotting, fig, ax, z_range, z_levels, x_range, y_range, 
                      cmap, x_label, y_label, z_label, title, axis_label)

def calc_difference_results(particles, other_particles, plotting = True, fig = None, ax = None, 
                            z_range = [None,None], z_levels = 1000, x_range = [None, None], y_range = [None, None], 
                            cmap = 'RdBu_r', x_label = "x / Arbitary Units", y_label ="y / Arbitary Units", 
                            z_label = "Potential Difference", title = "", axis_label = None):
    """
    Return the result of plotting the difference between particles and other particles
    Parameters
    ----------
    particles [Particle]: the particles to be analysed
    other_particles: [Particle]: the direct simulation particles
    ploting = True: Bool: whether or not to plot the particle
    fig: matplotlib figure object: the figure on which to plot the result if provided and plotting=True
    ax: matplotlib axes object: the axes on which to plot the result if provided and plotting=True
    z_range: [float, float]: the z range to be shown on a graph
    z_levels: Int: the number of z levels to be shown on the plot
    x_range: [float, float]: the x range to be plotted
    y_ragnge: [float, float]: the y range to be plotted
    cmap: String: the matplotlib colour map to use
    x_label: String: the x axis label
    y_label: String: the y axis label
    z_label: String: the z axis label
    title: String: the title of the graph
    axis_label: String: the axis label, e.g. a) in multi-subplot graphs
    Returns
    -------
    Results object
    """
    xs = []
    ys = []
    potential_differences = []
    for i, particle in enumerate(particles):
        position = particle.position
        xs.append(position[0])
        ys.append(position[1])
        
        potential_differences.append(particle.total_potential - other_particles[i].total_potential)
    
    return analyse_2D(xs, ys, potential_differences, plotting, fig, ax, z_range, z_levels, x_range, 
                      y_range, cmap, x_label, y_label, z_label, title, axis_label)
    
def calc_log_relative_error_results(particles, reference_particles, plotting = True, fig = None, 
                                    ax = None, z_range = [None, None], z_levels = 1000, x_range = [None, None], 
                                    y_range = [None, None], cmap = 'jet', x_label = "x / Arbitary Units", y_label ="y / Arbitary Units", 
                                    z_label = r"$\log_{10}(\text{Potential Difference})$", title = "", axis_label = None):
    """
    Return the result from finding the logarithm of the relative difference between particles and reference_particles
    Parameters
    ----------
    particles [Particle]: the simulated particles
    reference_particles [Particle]: the reference particles
    ploting = True: Bool: whether or not to plot the particle
    fig: matplotlib figure object: the figure on which to plot the result if provided and plotting=True
    ax: matplotlib axes object: the axes on which to plot the result if provided and plotting=True
    z_range: [float, float]: the z range to be shown on a graph
    z_levels: Int: the number of z levels to be shown on the plot
    x_range: [float, float]: the x range to be plotted
    y_ragnge: [float, float]: the y range to be plotted
    cmap: String: the matplotlib colour map to use
    x_label: String: the x axis label
    y_label: String: the y axis label
    z_label: String: the z axis label
    title: String: the title of the graph
    axis_label: String: the axis label, e.g. a) in multi-subplot graphs
    Returns
    -------
    Results object
    """
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
    return analyse_2D(xs, ys, log_rel_errors, plotting, fig, ax, z_range, z_levels, x_range, y_range, cmap, 
                      x_label, y_label, z_label, title, axis_label)

def calc_3D_results(particles, centre_position, plotting = True, fig = None, 
                    ax = None, scatter = False, marker_size = 1, x_range = [None, None], 
                    y_range = [None, None], x_label = "x / Arbitary Units", y_label ="y / Arbitary Units", 
                    title = "", label = "", format = "", legend = False, axis_label = None):
    """
    Calculates the results for particles in 3D space
    Parameters
    ----------
    particles [Particle]: the particles to be plotted
    centre_position: [Float]: the position of the centre of the 3D volume
    ploting = True: Bool: whether or not to plot the particle
    fig: matplotlib figure object: the figure on which to plot the result if provided and plotting=True
    ax: matplotlib axes object: the axes on which to plot the result if provided and plotting=True
    marker_size: Float: the size of scatter markers
    x_range: [float, float]: the x range to be plotted
    y_range: [float, float]: the y range to be plotted
    x_label: String: the x axis label
    y_label: String: the y axis label
    title: String: the title of the graph
    label: String: the name to label the plot with
    format: String: the format for the matplotlib sacatter plot
    legend: Bool: whether or not a legend is shown
    axis_label: String: the axis label, e.g. a) in multi-subplot graphs
    Returns
    -------
    Results object
    """
    rs = []
    accelerations = []
    for particle in particles:
        r = np.linalg.norm(particle.position-centre_position)
        if r!=0:
            rs.append(r)
            accelerations.append(np.linalg.norm(particle.acceleration))
    return analyse_3D(rs, accelerations, plotting, fig, ax, scatter, marker_size, x_range, y_range, x_label, y_label, 
                      title, label, format, legend, axis_label)

def calc_3D_difference_results(particles, other_particles, centre_position, plotting = True, 
                               fig = None, ax = None, scatter = False, marker_size = 1, x_range = [None, None], y_range = [None, None], 
                               x_label = "x / Arbitary Units", y_label ="y / Arbitary Units", title = "", label = "", format = "", 
                               legend = False, axis_label = None):
    """
    Calculates the difference in forces particles in 3D space
    Parameters
    ----------
    particles [Particle]: the particles to be plotted
    other_particles: [Particles]: the other particles to be compared to
    centre_position: [Float]: the position of the centre of the 3D volume
    ploting = True: Bool: whether or not to plot the particle
    fig: matplotlib figure object: the figure on which to plot the result if provided and plotting=True
    ax: matplotlib axes object: the axes on which to plot the result if provided and plotting=True
    marker_size: Float: the size of scatter markers
    x_range: [float, float]: the x range to be plotted
    y_range: [float, float]: the y range to be plotted
    x_label: String: the x axis label
    y_label: String: the y axis label
    title: String: the title of the graph
    label: String: the name to label the plot with
    format: String: the format for the matplotlib sacatter plot
    legend: Bool: whether or not a legend is shown
    axis_label: String: the axis label, e.g. a) in multi-subplot graphs
    Returns
    -------
    Results object
    """
    rs = []
    delta_accelerations = []
    for i, particle in enumerate(particles):
        r = np.linalg.norm(particle.position-centre_position)
        if r != 0:
            rs.append(r)
            delta_accelerations.append(np.linalg.norm(particle.acceleration - other_particles[i].acceleration))
    
    return analyse_3D(rs, delta_accelerations, plotting, fig, ax, scatter, marker_size, x_range, y_range, x_label, 
                      y_label, title, label, format, legend, axis_label)
    
def calc_3D_relative_error_results(particles, reference_particles, centre_position, plotting = True, 
                                   fig = None, ax = None, scatter = False, marker_size = 1, x_range = [None, None], 
                                   y_range = [None, None], x_label = "x / Arbitary Units", y_label ="y / Arbitary Units", 
                                   title = "", label = "", format = "", legend = False, axis_label = None):
    """
    Calculates the relative errors in forces particles in 3D space
    Parameters
    ----------
    particles [Particle]: the particles to be plotted
    reference_particles: [Particles]: the other particles to be compared to
    centre_position: [Float]: the position of the centre of the 3D volume
    ploting = True: Bool: whether or not to plot the particle
    fig: matplotlib figure object: the figure on which to plot the result if provided and plotting=True
    ax: matplotlib axes object: the axes on which to plot the result if provided and plotting=True
    marker_size: Float: the size of scatter markers
    x_range: [float, float]: the x range to be plotted
    y_range: [float, float]: the y range to be plotted
    x_label: String: the x axis label
    y_label: String: the y axis label
    title: String: the title of the graph
    label: String: the name to label the plot with
    format: String: the format for the matplotlib sacatter plot
    legend: Bool: whether or not a legend is shown
    axis_label: String: the axis label, e.g. a) in multi-subplot graphs
    Returns
    -------
    Results object
    """
    rs = []
    delta_accelerations = []
    reference_accelerations = []
    for i, particle in enumerate(particles):
        r = np.linalg.norm(particle.position-centre_position)
        if r != 0:
            rs.append(r)
            delta_accelerations.append(np.linalg.norm(particle.acceleration - reference_particles[i].acceleration))
            reference_accelerations.append(np.linalg.norm(reference_particles[i].acceleration))
    log_rel_errors = np.log10(np.abs(np.array(delta_accelerations) / np.array(reference_accelerations)))

    return analyse_3D(rs, log_rel_errors, plotting, fig, ax, scatter, marker_size, x_range, y_range, x_label, y_label, 
                      title, label, format, legend, axis_label)



def analyse_2D(xs, ys, potentials, plotting = False, fig = None, ax = None, z_range = [None, None], z_levels = 1000, x_range = [None, None], 
               y_range = [None, None], cmap = "jet", x_label = "x / Arbitary Units", y_label = "y / Arbitary Units", z_label = "z", title = "", axis_label = None):
    """Performs analysis of an x, y and z array passed in. Displays 2D tricontour
    graph depicting the points and returns a results object with the properties of this
    analysis"""
    keep_indices = np.invert(np.isinf(potentials) + np.isnan(potentials) + np.isinf(ys) + np.isnan(ys) + np.isinf(xs) + np.isnan(xs))
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
    

    
    if plotting:
        plt.ioff()
        if ax == None or fig == None:
            fig, ax = plt.subplots()
        
        # Plot to ax if ax provided, or otherwise plot noisily
        triang = tri.Triangulation(xs, ys)
        cntr = ax.tricontourf(triang, potentials, levels=levels, cmap=cmap)
        #ax.tricontour(triang, potential_differences, levels=4, colors='k')
        
        if x_range[0] != None:
            ax.set_xlim(left=x_range[0])
        if x_range[1] != None:
            ax.set_xlim(right=x_range[1])
        if y_range[0] != None:
            ax.set_ylim(bottom=y_range[0])
        if y_range[1] != None:
            ax.set_ylim(top=y_range[1])
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if z_range[0] != None and z_range[1] != None:
            ticks = np.linspace(z_range[0], z_range[1], 11)
            cbar = fig.colorbar(cntr, ax=ax, ticks=ticks)
        else:
            cbar = fig.colorbar(cntr, ax=ax)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(z_label, rotation=270)
        if title != "":
            ax.set_title(title, fontstyle='italic')
        if axis_label != None:
            ax.set_title(axis_label, fontfamily='serif', loc='left', fontsize='medium')
        plt.ion()

    return Results(np.max(potentials, initial=-20), np.min(potentials, initial=-20), np.mean(potentials), np.median(potentials), fig, ax)

def analyse_3D(xs, values, plotting = False, fig = None, ax = None, scatter = True, marker_size = 1, x_range = [None, None], y_range = [None, None], 
               x_label = "x / Arbitary Units", y_label = "y / Arbitary Units", title = "", label = "", format = "", legend = False, axis_label = None):
    """Performs analysis of x and values arrays passed in. Displays 2D graphs of 
    the arrays and returns a results object with the properties of the analysis"""
    keep_indices = np.invert(np.isinf(values) + np.isnan(values))
    values = np.array(values)[keep_indices]
    xs = np.array(xs)[keep_indices]
    xs_sort = xs.argsort()
    xs =  xs[xs_sort]
    values = values[xs_sort]

    if plotting:
        plt.ioff()
        if ax == None or fig == None:
            fig, ax = plt.subplots()
        
        # Plot to ax if ax provided, or otherwise plot noisily
        if scatter:
            ax.scatter(xs, values, s=marker_size, marker=format, label=label)
        else:
            ax.plot(xs, values, format, label=label)
        
        if x_range[0] != None:
            ax.set_xlim(left=x_range[0])
        if x_range[1] != None:
            ax.set_xlim(right=x_range[1])
        if y_range[0] != None:
            ax.set_ylim(bottom=y_range[0])
        if y_range[1] != None:
            ax.set_ylim(top=y_range[1])
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title != "":
            ax.set_title(title, fontstyle='italic')
        if axis_label != None:
            ax.set_title(axis_label, fontfamily='serif', loc='left', fontsize='medium')
        
        if legend:
            ax.legend()
        if abs(np.mean(values)) > 100 or abs(np.mean(values)) < 1E-2:
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        plt.ion()
    
    return Results(np.max(values, initial=-20), np.min(values, initial=-20), np.mean(values), np.median(values), fig, ax)

