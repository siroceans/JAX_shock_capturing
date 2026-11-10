import yaml
from solver.flux_functions import hll, steger_warming, hllc

def get_geometry(): 
    with open('config/geometry.yaml', 'r') as file: 
        config = yaml.safe_load(file)

    # renaming all variables like solver expects them
    L = config['geometry']['L']
    x_d = config['geometry']['discontinuity_x']
    n_x = config['geometry']['cell_number']

    rho_right = config['right_state']['rho']
    rho_left = config['left_state']['rho']

    p_right = config['right_state']['P']
    p_left = config['left_state']['P']

    gamma = config['gamma']
    u_0 = config['u_initial']

    # getting rid of intermediate variables in the scope
    del config 
    del file

    geometry_parameters = {k: v for k, v in locals().items()}
    return geometry_parameters 

def get_solver_config():
    with open('config/solver.yaml', 'r') as file: 
        config = yaml.safe_load(file)

    # renaming variables as expected...
    dt = config['dt']
    to = config['initial_time']
    t_f = config['final_time']
    CFL = config['CFL']

    # choosing solver
    if config['flux_solver'] == "steger_warming": 
        flux_solver = steger_warming
    elif config['flux_solver'] == "hll": 
        flux_solver = hll
    elif config['flux_solver'] == "hllc":
        flux_solver = hllc
    else: 
        raise ValueError("Flux Solver selected is not supported.")

    # getting rid of intermediate variables in scope
    del config
    del file

    solver_parameters = {k: v for k, v in locals().items()}
    return solver_parameters

def get_plotting_config():
    with open('config/plotting.yaml', 'r') as file: 
        config = yaml.safe_load(file)

    animation_time = config['animation_time']

    # getting rid of intermediate variables in scope
    del config
    del file

    #plotting_parameters = {k: v for k, v in locals().items()}
    return animation_time 

def get_training_config():
