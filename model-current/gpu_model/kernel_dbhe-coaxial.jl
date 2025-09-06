using Pkg
Pkg.activate(@__DIR__)
# Pkg.instantiate() # if everything works as expected, only run this and not "Pkg.add(...)"

# add more packages if needed here 
# Pkg.add("OrdinaryDiffEqTsit5")
# Pkg.add("OrdinaryDiffEqLowStorageRK")
# Pkg.add("DiffEqCallbacks")
# Pkg.add("Plots")
# Pkg.add("BenchmarkTools")
# Pkg.add("KernelAbstractions")
# Pkg.add("Adapt")
# Pkg.add("CUDA")
# Pkg.add("AMDGPU")


using OrdinaryDiffEqTsit5
using OrdinaryDiffEqLowStorageRK
using DiffEqCallbacks
using Plots
using BenchmarkTools
using KernelAbstractions
using Adapt
using CUDA
using AMDGPU

################################################################################
# Diffusion convection equation for temperature in a pipe-in-pipe geometry
#
# Equation: ρ c ∂ϕ/∂t =  ∂(λ ∂ϕ/∂x)/∂x + ∂(λ ∂ϕ/∂y)/∂y + ∂(λ ∂ϕ/∂z)/∂z
#                        -(ε vx ∂ϕ/∂x + ε vy ∂ϕ/∂y + ε vz ∂ϕ/∂z)
#                        + S
# Initial conditions:
#
# Boundary conditions:
#

# Physical parameters ##########################################################

# Maximum simulation time [s]
tspan = (0.0, 24.0) # 40 minutes

# Earth surface temperature [°C]
ϕs = 20


# Rock: granite #########################################
# Rock density [g/m3]
ρr = 2750000
# Rock specific heat [J/(g °C)]
cr = 0.790
# Rock thermal conductivity [W/(m °C)]
λr = 2.62
# Rock diffusion coefficient
dr = λr/(ρr*cr)
# Rock temperature as a function of depth [°C]
ϕ0_open(d, ϕs) = ϕs+0.5*d
ϕ0(d) = ϕ0_open(d, ϕs)

# Pipes: polyethylene ####################################
# Inner pipe inside radius [m]
r1 = 0.1
# Inner pipe thickness [m]
t1 = 0.01
# Inner pipe height [m]
h1 = 8.5
# Outer pipe inside radius [m]
vol1 = π*r1^2*h1
r2 = sqrt((vol1 + π*(r1+t1)^2*h1)/(h1*π)) # <= vol1 = vol2 = π*r2^2*h1-π*(r1+t1)^2*h1
# Outer pipe thickness [m]
t2 = 0.01
# Outer pipe height [m]
h2 = 9
# Porosity: ratio of liquid volume to the total volume
ε = 1
# Pipe density [g/m3]
ρp = 961000 
# Pipe specific heat [J/(g °C)]
cp = 2.9
# Pipe thermal conductivity [W/(m °C)]
λp = 0.54 
# Pipe diffusion coefficient
dp = λp/(ρp*cp)

# Fluid: water #########################################
# Fluid density [g/m3]
ρf = 997000
# Fluid specific heat capacity [J/(g °C)]
cf = 4.184 
# Fluid thermal conductivity [W/(m °C)]
λf = 0.6
# Fluid diffusion coefficient
df = λf/(ρf*cf)
# Flow speed [m/s]
uf = 0.01
vx0 = uf
vy0 = 0
vz0 = 0
# Characteristic linear dimension (diameter of the pipe) [m]
Lf = 2r1
# Fluid dynamic viscosity at 25 °C [Pa⋅s]
μf = 0.00089 # 0.0005465 at 50 °C
# Reynolds number
Re = ρf*uf*Lf/μf

# See https://gchem.cm.utexas.edu/data/section2.php?target=heat-capacities.php
#     https://en.wikipedia.org/wiki/High-density_polyethylene
#     https://en.wikipedia.org/wiki/Numerical_solution_of_the_convection%E2%80%93diffusion_equation

# Geometry distances [m]
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
zmin, zmax = 0.0, h2+1

Nx = 101
Ny = 101
Nz = 81

gridx = range(xmin, xmax, length=Nx)
gridy = range(ymin, ymax, length=Ny)
gridz = range(zmin, zmax, length=Nz)

ϕ = zeros(Nx, Ny, Nz)