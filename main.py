from solver.solver import shockTubeSolver
import solver.plotting as plotting
from config.config import get_geometry, get_solver_config, get_plotting_config

if __name__ == "__main__":
    # Testing the solver
    geometry_parameters = get_geometry()
    solver_parameters = get_solver_config()
    animation_time = get_plotting_config()
    u, rho, p = shockTubeSolver(**solver_parameters, **geometry_parameters)

    # plotting last state or animating all
    #plotting.animate_states(p, solver_parameters['L'], "P", animation_time)
    plotting.plot_final_state(u, geometry_parameters['L'], "U")