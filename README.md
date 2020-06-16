conduit-wave-benchmark

Author: Aaron Stubblefield (Columbia University, LDEO).

# Overview
This repository contains FEniCS python code for testing a finite element model for fluid flow through
deformable conduits against solitary wave solutions.
Details about the model and relevant analysis are in the paper:

>Stubblefield, A. G., Spiegelman, M., & Creyts, T. T. (2020). *Solitary waves in
power-law deformable conduits with laminar or turbulent fluid flow*. Journal of
Fluid Mechanics, 886.

The model equations and error calculation are also summarized in the comments at the
top of **conduit_verify.py**.

# Dependencies
This code runs with the current FEniCS Docker image (https://fenicsproject.org/download/).
Docker may be obtained at: https://www.docker.com/. To run the Docker image:

`docker run -ti -p 127.0.0.1:8000:8000 -v $(pwd):/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable:current`

The program also utilizes an ODE solver from the latest version of SciPy.
In the Docker container, SciPy must be updated via `pip3 install --upgrade scipy`.

# Running the code
To run the code:

1. Start the FEniCS Docker image.

2. Upgrade SciPy: `pip3 install --upgrade scipy`

3. Run `python3 conduit_verify.py`

This runs the code with the default options.
Read below for a summary of model options.

A png called 'verify_fig' is saved in the *results* directory upon completion,
showing the relative wave speed and shape errors over time.

# Output
Model output is saved in a *results* directory. This includes:
1. *vtkfiles* subdirectory containing vtk files of the solution at each
timestep. These may be viewed in ParaView (https://www.paraview.org/).
2. time coordinate `t`
3. spatial coordinate `x`
4. initial conduit profile `S_0`
5. final (computed) profile `S_T`
6. final (true) profile `Strue_T`
7. wave-speed error `delta_c` at specified time increments.
8. wave shape error `L2err` at specified time increments.

For explanation of the error calculation, refer to the paper or the
comments at the top of **conduit_verify.py**.

# Model options

Model parameters are set on the command line.
To see all model options, run `python3 conduit_verify.py -h`.
Parameters of interest include the stress exponent `n`, wave speed `c`,
and the flow law parameters `alpha` and `beta`.
