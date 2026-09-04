import yaml

def get_parameters(): 
    with open('input_parameters.yaml', 'r') as file: 
        config = yaml.safe_load(file)

    # renaming all variables like solver expects them
    L = config['geometry']['L']
    x_discontinuity = config['geometry']['discontinuity_x']
    n_x = config['geometry']['cell_number']

    rho_right = config['right_state']['rho']
    rho_left = config['left_state']['rho']

    P_right = config['right_state']['P']
    P_left = config['left_state']['P']

    gamma = config['gamma']
    u_0 = config['u_initial']
    t_f = config['final_time']
    CFL = config['CFL']

    # getting rid of intermediate variables in the scope
    del config 
    del file
    return locals() # returning all variables in the scope as a dictionary
