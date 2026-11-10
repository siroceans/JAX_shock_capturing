from solver.solver import shockTubeSolver
import solver.plotting as plotting
import config.config as cf

def test_solver():
    # Testing the solver
    geometry_parameters = cf.get_geometry()
    solver_parameters = cf.get_solver_config()
    animation_time = cf.get_plotting_config()
    u, rho, p = shockTubeSolver(**solver_parameters, **geometry_parameters)

    # plotting last state or animating all
    #plotting.animate_states(p, solver_parameters['L'], "P", animation_time)
    plotting.plot_final_state(u, geometry_parameters['L'], "U")

def compute_dt(L, nx, CFL): 
    dx = L / nx
    c_max = 2.2
    dt = CFL * dx / c_max
    return dt


if __name__ == "__main__":
    test_solver()
    #print(compute_dt(1, 1000, 0.5))