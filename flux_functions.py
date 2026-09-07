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
    u_l = u[0:-1]
    u_r = u[1:]
    a_l = a[0:-1]
    a_r = a[1:]

    # Initializing arrays to make sure they have right dimensions
    S_L = jnp.zeros((U.shape[0] - 1, 1))
    S_R = jnp.zeros((U.shape[0] - 1, 1))

    # filling in values.... 
    S_L = S_L.at[:, 0].set(jnp.minimum(u_l - a_l, u_r - a_r))
    S_R = S_R.at[:, 0].set(jnp.maximum(u_l + a_l, u_r + a_r))
    return S_L, S_R

def s_star(U, gamma, S_L, S_R): 
    # S_* estimate using (10.58) from Toro's textbook
    rho, u, _, _, p = flow_properties(U, gamma)

    S_star = (p[1:] - p[0:-1] + rho[0:-1] * u[0:-1] * (S_L - u[0:-1]) - 
              rho[1:] * u[1:] * (S_R - u[1:])) / (rho[0:-1] * (S_L - u[0:-1]) - 
                                                  rho[1:] * (S_R - u[1:]))
    return S_star

def compute_F(n_x, rho, u, a, gamma):
     # creating F array
    F = jnp.zeros((n_x, 3))
    F = F.at[:, 0].set(rho * u)
    F = F.at[:, 1].set(rho * u**2 + (rho * a**2)/gamma)
    F = F.at[:, 2].set(rho * ((a**2 * u)/(gamma - 1) + 1/2 * u**3))
    return F

def hll(U, gamma):
    # function that computes the hll fluxes
    rho, u, a, c, P = flow_properties(U, gamma)
    n_x = U.shape[0]
    F = compute_F(n_x, rho, u, a, gamma)

    # computing L and R values
    S_L, S_R = direct_wave_speed(U, gamma)
    F_L = F[0:-1, :]
    F_R = F[1:, :]
    U_L = U[0:-1,:]
    U_R = U[1:, :]

    # Computing F^hll flux
    F_hll = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L))/(S_R - S_L)

    # Choosing value for interfaces
    F_hll_i = jnp.where(S_L >= 0, F_L, 0)
    F_hll_i = jnp.where((S_L <= 0) & (S_R >= 0), F_hll, F_hll_i)
    F_hll_i = jnp.where(S_R <= 0, F_R, F_hll_i)

    F_p_h = F_hll_i[1:, :]
    F_m_h = F_hll_i[0:-1, :]
    return F_p_h, F_m_h

def hllc(U, gamma):
    # computing HLLC fluxes according to section 10.6.1 from Toro's textbook
    n_x = U.shape[0]
    rho, u, a, _, p = flow_properties(U, gamma)
    S_L, S_R = direct_wave_speed(U, gamma)
    S_s = s_star(U, gamma, S_L, S_R)

    # computing U_*R and U_*L using (10.33)
    U_sR = jnp.zeros((n_x, 3))
    U_sL = jnp.zeros((n_x, 3))

    factor_R = rho[1:] * (S_R - u[1:])/(S_R - S_s)
    U_sR = U_sR.at[:, 0].set(factor_R)
    U_sR = U_sR.at[:, 1].set(S_s * factor_R)
    U_sR = U_sR.at[:, 2].set((U[1:, 2]/rho[1:] + (S_s - u[1:]) * 
                             (S_s + p[1:]/(rho[1:] * (S_R - u[1:])))) * factor_R)

    factor_L = rho[0:-1] * (S_L - u[0:-1])/(S_L - S_s)
    U_sL = U_sL.at[:, 0].set(factor_L)
    U_sL = U_sL.at[:, 1].set(S_s * factor_L)
    U_sL = U_sL.at[:, 2].set((U[0:-1, 2]/rho[0:-1] + (S_s - u[0:-1]) * 
                             (S_s + p[0:-1]/(rho[0:-1] * (S_L - u[0:-1])))) * factor_L)

    # computing HLLC fluxes using (10.34)
    F = jnp