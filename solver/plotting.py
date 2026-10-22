import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_final_state(states, L, variable):
    n_x = states[0].shape[0]
    dx = L/n_x
    x = jnp.arange(0 + dx/2, L, dx) # array of centroid locations
    plt.plot(x, states[-1])
    plt.title("Final " + variable + " distribution.")
    plt.xlabel("x")
    plt.ylabel(variable)
    plt.show()

def create_plot(states, L): 
    n_x = states[0].shape[0]
    dx = L/n_x
    x = jnp.arange(0 + dx/2, L, dx)

    fig, ax = plt.subplots()
    line, = ax.plot(x, states[0])
    ax.set_ylim((states.min(), states.max()))
    return fig, ax, line

def animate_states(states, L, variable, seconds):
    fig, ax, line = create_plot(states, L)

    def animation_helper(frame):
        line.set_ydata(states[frame])

    speed = seconds * 100 / len(states)
    ani = animation.FuncAnimation(fig = fig, func = animation_helper, frames = len(states), 
                                  interval = speed)

    plt.title(variable + " evolution over time.")
    plt.xlabel("x")
    plt.ylabel("variable")
    plt.show()
