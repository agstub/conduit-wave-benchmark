"""
FEniCS program for computing cross-sectional area S, effective pressure N, discharge Q in
the coupled PDE system:

dS/dt = -dQ/dx
dQ/dx = SN|N|^(n-1)
Q = (S^alpha)*(1+dNdx)^beta

where alpha>1, beta>0, and n>=1 are parameters.

This program verifies the finite element solver for the above PDE system by
using a solitary wave solution initial condition. The solitary wave initial condition
solves the ODE system:

dS/dxi = (1/c)S*sgn(N)*|N|^n
dN/dxi = P(S,alpha,beta,c)

where c is the true wave speed, P is a function (see below), and xi = x-ct
is the travelling wave coordinate.

The program returns the error in wave speed |c-c_*|/c, where c_* mimimizes the L2-norm of the
error || S(x-c_* t) - S_a(x,t) ||; S_a is the "true" solitary wave solution computed
by a SciPy ODE solver.

More details can be found in the paper:

Stubblefield, A. G., Spiegelman, M., & Creyts, T. T. (2020). Solitary waves in
power-law deformable conduits with laminar or turbulent fluid flow. Journal of
Fluid Mechanics, 886.

"""
from __future__ import print_function
import sys,argparse

parser = argparse.ArgumentParser(\
    description='Solve the viscous conduit equations numerically with an "exact solution" solitary wave initial condition.')
parser.add_argument('-alpha', type=float, default=2.0, metavar='X',
                    help='Exponent on S')
parser.add_argument('-c', type=float, default=2.5, metavar='X',
                    help='Wave speed')
parser.add_argument('-beta', type=float, default=1.0, metavar='X',
                    help='Exponent on 1 + dNdx. Choose 1 or 0.5 only!')
parser.add_argument('-n', type=float, default=2.0, metavar='X',
                    help='Exponent on N|N|^n-1')
parser.add_argument('-length', type=float, default=2000.0, metavar='X',
                    help='Length of space domain [0,length]')
parser.add_argument('-T', type=float, default=500.0, metavar='X',
                    help='Length of time domain [0,T]')
parser.add_argument('-CFL', type=float, default=0.5, metavar='X',
                    help='CFL number: c*dt/dx = CFL, where c is the wave speed. ')
parser.add_argument('-nx', type=int, default=2000, metavar='X',
                    help='number of grid points')
parser.add_argument('-theta', type=float, default=0.5, metavar='X',
                    help='value used in theta-method.0 = backward euler, 0.5 = trapezoidal rule, 1 = forward euler')

args, unknown = parser.parse_known_args()

from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import scipy.integrate as scpI
import os

set_log_level(40)               # Suppress Newton convergence information if desired.

#-----------Functions for computing analytical solitary wave--------------------
def f(t,y,c,n,a,b):
    #Define RHS of the dynamical system
    S, N = y
    dydt = [(1.0/c)*S*(np.sign(N)*np.abs(N)**(n)), \
    (np.sign(c*S + (1.0-c))*np.abs(c*S + (1.0-c))**(1.0/b))*(S**(-a/b)) - 1]
    return dydt

def f1(S,c,n,a,b):
    # Function for finding the wave amplitude (i.e., root of Hamiltonian function)

    # turbulent case
    if np.abs(b - 0.5)<1.0e-4:
        C = hamT(1.0,0,a,c,n)
        H = hamT(S,0,a,c,n) - C

    # laminar case
    elif np.abs(b -1)<1.0e-4:
        C = hamL(1.0,0,a,c,n)
        H = hamL(S,0,a,c,n) - C

    return H

def Ip(S,a,c):
    # Auxiliary function for computing beta=1/2 (turbulent) Hamiltonian
    # Needed for calculating wave amplitude
 return (((-c**2)/(2*a-2))*(S**(2-2*a))+(-2*(1-c)*c/(2*a-1))*(S**(1-2*a)) \
         + (-((1-c)**2)/(2*a))*(S**(-2*a)))

def Hside(S,c):
    # Indicator function for computing beta=1/2 (turbulent) energy function
    # Needed for calculating wave amplitude
    if np.size(S)>1:
        q = np.zeros(np.shape(S))
        q[S<1.0-1.0/c] = 1.0
    if np.size(S) == 1:
        q = 0
        if S< 1.0-1.0/c:
            q = 1
    return q

def hamT(S,N,a,c,n):
    # Definition of hamiltonian for beta = 1/2
    # Needed for calculating wave amplitude
    s0 = np.sign(c*S+1-c)
    HT= (-(np.abs(N)**(n+1))/(c*(n+1)) + s0*Ip(S,a,c) +2*Ip(1.0-1.0/c,a,c)*Hside(S,c) - np.log(S))
    return HT

def hamL(S,N,a,c,n):
    # Definition of hamiltonian for beta = 1
    # Needed for calculating wave amplitude
    HL = -(np.abs(N)**(n+1))/(c*(n+1))-(c*(S**(1-a)))/(a-1)-((1-c)*(S**(-a)))/a-np.log(S)
    return HL

def solwaves(L,a,c,n,b):
    # ODE solver for the solitary wave solutions
    # L is length of the domain

    tspan = (0,L)           # integration interval

    r0 = 2.0                # Initial estimate of wave amplitude

    # Calculate wave amplitude by finding root of hamiltonian functions
    root = optimize.newton(f1,r0,args = (c,n,a,b))

    y0 = np.array([root,0])

    # solve the ODE for the solitary wave solution
    soln = scpI.solve_ivp(lambda t,y: f(t,y,c,n,a,b),tspan,y0,method='Radau',dense_output=True,max_step=0.005)

    return soln

#-------------------------------------------------------------------------------
def calcerror(u,x0):
    # Function for calculating the shape error and speed error of the solitary
    # wave solution
    res = optimize.minimize_scalar(lambda delta: errornorm(u.sub(0),\
            project(uexpr(element=V.ufl_element(),domain=mesh,x0=x0+delta),V).sub(0),mesh=mesh) )

    # mismatch in true vs. computed wave speed
    delta = res.x

    # shape error
    L2err = errornorm(u.sub(0),project(uexpr(element=V.ufl_element(), \
            domain=mesh,x0=x0+delta),V).sub(0),mesh=mesh)/norm(project(u.sub(0)-1,W))

    return delta,L2err

#-------------------------------------------------------------------------------
# Define Dirichlet part of the boundary
# Used to approximate the condition (S,N,Q) -> (1,0,1) as |x| -> infty.
def boundary(x, on_boundary):
    return on_boundary

#-------------------------------------------------------------------------------

class uexpr(UserExpression):
    # FEniCS expression for "analytical" solitary wave
    def __init__(self, x0, **kwargs):
        self.x0 = x0
        super().__init__(**kwargs)
    def eval(self,value,x):
        # Convert coordinates and evaluate analytical solution
        u_sol = sol.__call__(np.abs(x[0]-self.x0))
        value[0] = u_sol[0]                             # S value
        value[1] = u_sol[1]*np.sign(x[0]-self.x0)       # N value
        value[2] = c*u_sol[0] + 1-c                     # Q value

#-------------------------------------------------------------------------------

os.mkdir('results')   # Make a directory for the results.


# Define model parameters
T = args.T                                      # final time
length = args.length                            # length of interval
c = args.c                                      # wave speed
nx = args.nx                                    # number of nodes
nt = int((1.0/args.CFL)*(c*T/length)*nx)        # number of time steps
dt = T / nt                                     # time step size

alpha = args.alpha                              # Exponent on S
beta = args.beta                                # Exponent on (1+dNdx)
n = args.n                                      # Stress Exponent.

theta = Constant(args.theta)                    # theta-method parameter:
                                                # 1 = Backward Euler,
                                                # 0.5 = Trapezoidal,
                                                # 0 = Forward Euler.

xi0 = length/4.0                                # Location of peak at t=0.

tplt = np.linspace(0,T,num=nt)                  # time array for plotting


err = np.zeros(nt)                              # array for storing shape errors
delta = np.zeros(nt)                            # array for storing speed errors

mesh = IntervalMesh(nx,0,length)                # define mesh

xarr = mesh.coordinates().flatten()             # x array for plotting

np.savetxt('results/x',xarr)                    # save x coordinate
np.savetxt('results/t',tplt)                    # save t coordinate

# Define finite elements and function spaces
P1 = FiniteElement('P',interval,1)
P2 = FiniteElement('P',interval,2)

element = MixedElement([P2,P1,P2])
V = FunctionSpace(mesh, element)

W = FunctionSpace(mesh,'P',2)

# Define test functions
S_t, N_t,Q_t = TestFunctions(V)


# Define function u that contains cross section S, effective pressure N, and discharge Q.
u = Function(V)                                 # function for current timestep
u_n = Function(V)                               # function for previous timestep

S, N, Q = split(u)
S_n, N_n, Q_n = split(u_n)


# set initial condition for true solution
print('Computing true solitary wave solution for initial condition...')
soln = solwaves(length,alpha,c,n,beta)
sol = soln.sol

t = 0                                          # time

# Create initial condition from
u0 = uexpr(element=V.ufl_element(),domain=mesh,x0=xi0)
u0 = project(u0,V)

 # save txt file of initial condition
S_0 = project(u0.sub(0),W).compute_vertex_values(mesh)
np.savetxt('results/S_0',project(u0.sub(0),W).compute_vertex_values(mesh))

# Assign initial condition to functions
u_n.assign(project(u0,V))
u.assign(project(u0,V))

# Define boundary conditions
bc = DirichletBC(V, (Constant(1.0),Constant(0.0),Constant(1.0)), boundary)

# Initialize Constants used in variational forms, defined above
k = Constant(dt)
alpha = Constant(alpha)
n = Constant(n)
beta = Constant(beta)

# Define functions for time-stepping
S_theta = theta*S+(1-theta)*S_n
N_theta = theta*N+(1-theta)*N_n
Q_theta = theta*Q+(1-theta)*Q_n

# Define variational problem
Psi = (1.0 + Dx(N,0))

LS = (S - S_n)*S_t*dx  + k*(S_theta)*(N_theta*abs(N_theta)**(n-1))*S_t*dx

LM = -Dx(N_t,0)*(Q)*dx - S*((N*abs(N)**(n-1)))*N_t*dx + (Q)*N_t*ds

LQ = (Q-(S**(alpha))*(((Psi)*((abs(Psi))**(beta-1.0)))))*Q_t*dx

L = LS + LM + LQ
#------------------------------------------------------------------

# Create VTK files
vtkfile_S = File('results/vtkfiles/S.pvd')
vtkfile_N = File('results/vtkfiles/N.pvd')

# Time-stepping
for i in range(nt):
    print("timestep "+str(i+1)+' out of '+str(nt))

    # Update current time
    t += dt

    # Solve variational problem for time step
    solve(L == 0, u,bcs=bc,solver_parameters={"newton_solver":{"relative_tolerance":1e-13},"newton_solver":{"maximum_iterations":50}})
    x0 = xi0 + c*t

    #Save VTK files
    _S, _N, _Q = u.split()
    vtkfile_S << (_S, t)
    vtkfile_N << (_N, t)

    # Compute error in increments of t = 20
    if t%20 < dt:
        print('Computing error at time t = '+str(int(t)))
        x0 = xi0 + c*t

        d,L2err = calcerror(u,x0)

        err[i] = L2err                      # Relative shape error
        delta[i] = np.abs(d/(c*t))          # Relative speed error

    # Update previous solution
    u_n.assign(u)


# Save shape error and speed error
np.savetxt('results/L2err',err)
np.savetxt('results/delta_c',delta)


# Compute final time solution
ua = project(uexpr(element=V.ufl_element(),domain=mesh,x0=x0),V)

S_T = project(u.sub(0),W).compute_vertex_values(mesh)

Strue_T = project(ua.sub(0),W).compute_vertex_values(mesh)

# save txt file of final time solution and true solution
np.savetxt('results/S_T',S_T)
np.savetxt('results/Strue_T',Strue_T)


# Plotting
plt.figure(figsize=(10,10))
plt.subplot(311)
plt.annotate(r'$t=0$',xy=(250,1.6),fontsize=20,bbox=dict(facecolor='aliceblue', alpha=1))
plt.annotate(r'$t=T$',xy=(1850,1.6),fontsize=20,bbox=dict(facecolor='aliceblue', alpha=1))
plt.plot(xarr[xarr<length/2],S_0[xarr<length/2],color='crimson',linewidth=2,label=r'initial')
plt.plot(xarr[xarr>=length/2],S_T[xarr>=length/2],color='slateblue',linewidth=2,label=r'computed')
plt.plot(xarr[xarr>=length/2],Strue_T[xarr>=length/2],color='k',linewidth=2,linestyle='--',label=r'true')
plt.xlabel(r'$x$',fontsize=24)
plt.ylabel(r'$S$',fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16,loc='lower left',framealpha=0.1)

plt.subplot(312)
plt.plot(tplt[err>0]/500.0,err[err>0],'ko',markersize=10,label=r'$(\alpha,\beta,n)=(2,1,2)$')
plt.plot(tplt[err>0]/500.0,err[err>0],'k-',linewidth=2)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.gca().yaxis.offsetText.set_fontsize(20)
plt.gca().yaxis.major.formatter._useMathText = True
plt.xticks(fontsize=0)
plt.yticks(fontsize=20)
plt.ylabel(r'$\frac{\Vert e \Vert_{L^2}}{\Vert S_h - 1 \Vert_{L^2}}$',fontsize=30)

plt.subplot(313)
plt.plot(tplt[delta>0]/500.0,delta[delta>0],'ko',markersize=10)
plt.plot(tplt[delta>0]/500.0,delta[delta>0],'k-',linewidth=2)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.gca().yaxis.offsetText.set_fontsize(20)
plt.gca().yaxis.major.formatter._useMathText = True
plt.xticks(fontsize=16)
plt.yticks(fontsize=20)
plt.ylabel(r'$\frac{|c-c_\star|}{c}$',fontsize=30)
plt.xlabel(r'$t\,/\,T$',fontsize=22)
plt.tight_layout()
plt.savefig('results/verify_fig')
