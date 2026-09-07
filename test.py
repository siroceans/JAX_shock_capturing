from solver import shockTubeSolver
import plotting
from solver_config import get_parameters

if __name__ == "__main__":
    # Testing the solver
    solver_parameters, animation_time = get_parameters()
    u, rho, p = shockTubeSolver(**solver_parameters)
    plotting.animate_states(p, solver_parameters['L'], "P", animation_time)
    