# Project Weekly Updates
*summer updates and progress report presented in a separate document*  

## Weeks of 08/30/2026 -> 09/12/2026
During these couple of weeks, the following things were done:  
- Wrote a Steger-Warming flux solver. 
- Wrote an HLL flux solver. 
- Wrote an HLLC flux solver. 

In addition to the flux solvers, a main loop was written to solve the equations through time.
A first order explicit Euler approach was chosen for this portion of the code.
All of the fluxes were evaluated using a first order scheme, that is, no interpolation scheme was used.  
  
$U_{i}^{n+1}=U_{i}^{n}+ (\Delta t)/(\Delta x) [F_{i - 1/2} - F_{i + 1/2}]$
  
The following considerations were made for the architectural choices of the program:  
    - [x] Flux solver modularity: the program must have the ability to interchangeably choose which approach it will use to solve for the fluxes.  
    - [x] All the functions written must be able to be vectorized and jit comiled using JAX.  
    - [x] Solver parameters should all be contained in a separate file, as to allow for easier training of the models in the future.   


## Week of 09/13/2026 to 09/18/2026  
The next step for this project is adding the neural network (NN) architecture on top of each flux solver during each of the iteration steps.
JAX does not natively offer support for networks and optimization for these; however, they list multiple frameworks that have been developed for use with JAX.
After reading the documentation for each of them, I decided to use **Equinox** rather than Flax, since I liked its philosophy behind their design more. 

The first goal for this step of the project is to *write a simple MLP on top of each of the flux computations*. Before tackling this, I need to do the following:  
    - [ ] Learn Equinox: this will be achieved by following its documentation to do the basic MNIST MLP example. 


