import yaml
from flux_functions import hll, steger_warming, hllc

def get_parameters(): 
    with open('input_parameters.yaml', 'r') as file: 
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

    animation_time = config['animation_time']

    not_solver_parameters = ['not_solver_parameters', 'animation_time']

    # getting rid of intermediate variables in the scope
    del config 
    del file

    solver_parameters = {k: v for k, v in locals().items() if k not in not_solver_parameters}

    return solver_parameters, animation_time
