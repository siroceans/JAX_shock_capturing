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

    # evaluating fluxes at cell faces (excluding the boudnaries)
    F_p_h = F_p[1:-1, :] + F_n[2:, :]   #F_i+1/2
    F_n_h = F_p[0:-2, :] + F_n[1:-1, :] #F_i-1/2
    return F_p_h, F_n_h

def direct_wave_speed(U, gamma):
    # direct wave speed estimate using section 10.5.1 from Toro's textbook as a reference
    # using 10.38 as reference!
    # also NOT doing any reconstruction. TODO: reconstruction step has to be implemented here too!

    # ignoring boundary cells...
    _, u, a, _, _ = flow_properties(U, gamma)
    u_l = u[0:-2]
    u_r = u[2:]
    a_l = a[0:-2]
    a_r = a[2:]

    S_L = jnp.minimum(u_l - a_l, u_r - a_r)
    S_R = jnp.maximum(u_l + a_l, u_r + a_r)
    return S_L, S_R

def hll(U, gamma):
    # function that computes the hll fluxes
    rho, u, a, c, P = flow_properties(U, gamma)
    n_x = U.shape[0]

    # creating F array
    F = jnp.zeros((n_x, 3))
    F = F.at[:, 0].set(rho * u)
    F = F.at[:, 1].set(rho * u**2 + (rho * a**2)/gamma)
    F = F.at[:, 2].set(rho * ((a**2 * u)/(gamma - 1) + 1/2 * u**3))

    # computing wave speeds
    S_L, S_R = direct_wave_speed(U, gamma)

    # creating F^(hll) arrays
    F_hll = (S_R * F[0:-2, :] - S_L * F[2:, :] + S_L * S_R * (U[2:, :] - U[0:-2, :]))/(S_R - S_L)

    F_hll_interfaces = jnp.where(S_L >= 0, F[0:-2, :], 0)
    F_hll_interfaces = jnp.where((S_L <= 0) and (S_R >= 0), F_hll, F_hll_interfaces)
    F_hll_interfaces = jnp.where(S_R <= 0, F[2:, :],  F_hll_interfaces)

    F_p_h = F_hll_interfaces[1:, :]
    F_m_h = F_hll_interfaces[0:-1, :]
    return F_p_h, F_m_h

