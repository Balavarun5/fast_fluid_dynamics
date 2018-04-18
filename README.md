###Fast Fluid Dynamics###
Fast Fluid Dynamics is a computationally cheap way to evolve stable fluids. The
seminal
[paper](http://www.dgp.toronto.edu/people/stam/reality/Research/pdf/ns.pdf)
in this topic was written by Jos Stam. In this project, the lid driven cavity
problem which is a benchmark CFD problem is simulated using Fast Fluid Dynamics.
The implementation has been written in
[Arrayfire](https://github.com/arrayfire/arrayfire) a general purpose GPU
library as well as Octave/Matlab. The code is a modified version of another
[code](https://github.com/lukasbystricky/FastFluidDynamics) available on github.
The advection step which was the rate determining step has been completely
vectorized for faster evolution and the difference can be seen in the execution
time. Here's a [gif](octave_implementation/results/n_200.gif) obtained by evolving for
a grid size of 200x200 cells using the octave/matlab implementation.
