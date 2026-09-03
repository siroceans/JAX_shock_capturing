import jax
import jax.numpy as jnp

# setting default to double fpn
jax.config.update("jax_enable_x64", True)

# module containing all of the functions used for the flux calculations to solve shock tube problem

def flow_properties(U, gamma):
    rho = U[:, 0]
    u = U[:, 1] / U[:, 0]
    P = (gamma - 1) * (U[:, 2] - 1/2 * (U[:, 1])**2 / U[:, 0])
    a = jnp.sqrt(P * gamma / rho)
    c = jnp.abs(u) + a

    return rho, u, a, c, P


def steger_warming(U, gamma):
    n_x = U.shape[0]
    rho, u, a, *_ = flow_properties(U, gamma)

    F_n = jnp.zeros((n_x, 3)) # negative component of the flux
    F_p = jnp.zeros((n_x, 3)) # positive component of the flux
    # defining multiplication factors for each component
    factor_n = 1/2 * rho / gamma * (u - a)
    factor_p = 1/2 * rho / gamma


    # computing negative F
    F_n = F_n.at[:, 0].set(1 * factor_n)
    F_n = F_n.at[:, 1].set((u - a) * factor_n)
    F_n = F_n.at[:, 2].set((1/2 * (u-a)**2 + 1/2 * a**2 * ((3-gamma) / (gamma-1))) * factor_n)

    # and then positive F...
    F_p = F_p.at[:, 0].set(((2 * gamma - 1) * u + a) * factor_p)
    F_p = F_p.at[:, 1].set((2 * (gamma-1) * u**2 + (u+a)**2) * factor_p)
    F_p = F_p.at[:, 2].set(((gamma-1) * u**3 + 1/2 * (u+a)**3 + 1/2 * a**2 * 
                           (3-gamma)/(gamma-1) * (u+a)) * factor_p)
    return F_p, F_n

