import jax
import jax.numpy as jnp
from flux_functions import steger_warming, flow_properties
import plotting

# setting default to double fpn
jax.config.update("jax_enable_x64", True)

## Solving Sod's shock tube problem using different flux approaches in FVM
# Jorge Luis Mares Zamora
# Last worked on:

def initializeField(rho_right, rho_left, P_right, P_left, x_discontinuity, n_x, gamma, L, u_0):
    dx = L / n_x # remember that nx is the no. of cells, not the no. of edges!!
    U = jnp.zeros((n_x, 3)) # leading axis is the no. of cells that we have!
    centroid_x = jnp.arange(0 + dx/2, L, dx) # array of centroid locations

    # creating rho, P, and a arrays
    rho = jnp.where(centroid_x < x_discontinuity, rho_left, rho_right)
    P = jnp.where(centroid_x < x_discontinuity, P_left, P_right)
    a = jnp.sqrt(P * gamma / rho)

    # filling in the U matrix
    U = U.at[:, 0].set(rho)
    U = U.at[:, 1].set(rho * u_0)
    U = U.at[:, 2].set(rho * (a**2/(gamma * (gamma-1)) + 1/2 * u_0**2))

    # creating maximum wave speed array! (for cfl condition)
    c = jnp.abs(u_0) + a
    return U, c, dx


def shockTubeSolver(L, x_d, n_x, rho_right, rho_left, p_right, p_left, gamma, u_0, t_f):
    # initialize field
    U, c, dx = initializeField(rho_right,rho_left, p_right, p_left, x_d, n_x, gamma, L, u_0)

    #-------------------------
    # time marching loop!!!
    #-------------------------

    # using CFL to calculate dt and initializing time loop
    CFL = 0.5
    dt = CFL*  dx / jnp.max(c)
    t = 0

    # initializing and filling in initial states TODO: this surely can be optimized?
    rho, u, _, c, P = flow_properties(U, gamma)
    u_states = [u]
    rho_states = [rho]
    P_states = [P]

    while t < t_f:
        t = t + dt
        F_p_h, F_n_h = steger_warming(U, gamma)

        # ignoring boundary cells... (look at BCs!!!)
        U = U.at[1:-1, :].set(U[1:-1, :] - dt/dx  * (F_p_h - F_n_h))

        # computing current flow parameters and saving states
        rho, u, _, c, P = flow_properties(U, gamma)
        u_states.append(u)
        rho_states.append(rho)
        P_states.append(P)

        # recomputing dt 
        dt = CFL * dx / jnp.max(c)

        # we want to stop EXACTLY at t_f, so we will limit the last iteration
        if (t + dt) > t_f: 
            dt = t_f - t
    return jnp.stack(u_states), jnp.stack(rho_states), jnp.stack(P_states)

# testing code!

u, rho, p = shockTubeSolver(1, 0.5, 100, 0.125, 1, 0.1, 1, 1.4, 0, 0.2)
plotting.animate_states(u, 1, "U", 5)