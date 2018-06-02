# Fast Fluid Dynamics

Fast Fluid Dynamics is a computationally cheap way to evolve stable fluids. The
seminal
[paper](http://www.dgp.toronto.edu/people/stam/reality/Research/pdf/ns.pdf)
in this topic was written by Jos Stam. In this project, the lid driven cavity
problem which is a benchmark CFD problem is simulated using Fast Fluid Dynamics.
The implementation has been written in
[Arrayfire](https://github.com/arrayfire/arrayfire) a general purpose GPU
library as well as Octave/Matlab. The code is a modified version of another
[code](https://github.com/lukasbystricky/FastFluidDynamics) available on github.
The code is completely vectorized and the difference can be seen in the execution
time. Here's a [gif](octave_implementation/results/n_200.gif) obtained by evolving for
a grid size of 200x200 cells using the octave/matlab implementation.


The values obtained using the code have been compared to verified values from
the paper by [Ghia, Ghia,
Shin](https://pdfs.semanticscholar.org/211b/45b6a06336a72ca064a6e59b14ebc520211c.pdf).
The particular comparison of the velocity of the fluid in the x direction for a
lid driven cavity can be seen here for a Reynolds number of 
[100](octave_implementation/results/u_profile_re_100.png)
and [400](octave_implementation/results/u_profile_re_400.png). These trends show
that the value is converging to the verified values as the grid size used for
the same lid decreases.


The aim of the thesis which was to develop a code which implements Fast Fluid Dynamics and
verifies the claim that FFD is indeed faster than CFD methods has been achieved. The code is
completely vectorized and the improvements made on the available code is starkly visible.
The pdf of the thesis is also uploaded, which details the work done,
specifically the improvements made in speed of the simulation.


Thanks.
