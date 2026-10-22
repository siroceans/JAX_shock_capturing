import jax
import jax.numpy as jnp

# setting default to double fpn
jax.config.update("jax_enable_x64", True)

def generate_data(resolution:int): 
