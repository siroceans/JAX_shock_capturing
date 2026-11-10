import os
import jax
import jax.numpy as jnp
from solver.solver import initializeField, time_step
from solver.flux_functions import flow_properties

# setting default to double fpn
jax.config.update("jax_enable_x64", True)

def generate_data(L, x_d, n_x, rho_right, rho_left, p_right, p_left, gamma, u_0, t_f, CFL,
                     flux_solver, dt, t_o, directory):

    # initialize field and check no of iterations
    U, _, dx = initializeField(rho_right, rho_left, p_right, p_left, x_d, n_x, gamma, L, u_0)
    t = t_o
    iterations = int(round((t_f - t_o) / dt))
    width = len(str(iterations))
    time_array = t_o + jnp.arange(iterations + 1) * dt

    # check if destination directory exists, if not, then create
    os.makedirs(f'./neural_flux/training_data/{directory}', exist_ok=True)

    # save initial state to memory
    jnp.save(f'./neural_flux/training_data/{directory}/{0:0{width}d}', U)
    jnp.save(f'./neural_flux/training_data/{directory}/time_array', time_array)

    for i in range(1,iterations+1):
        # computing and saving next iteration to memory
        U = time_step(U, gamma, dt, dx, flux_solver)
        jnp.save(f'./neural_flux/training_data/{directory}/{i:0{width}d}', U)
   
        