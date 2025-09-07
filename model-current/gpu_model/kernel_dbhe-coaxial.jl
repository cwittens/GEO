using Pkg
Pkg.activate(@__DIR__)
# Pkg.instantiate() # if everything works as expected, only run this and not "Pkg.add(...)"

# add more packages if needed here
# Pkg.add("WriteVTK")
# Pkg.add("OrdinaryDiffEqLowOrderRK")
# Pkg.add("OrdinaryDiffEqTsit5")
# Pkg.add("OrdinaryDiffEqLowStorageRK")
# Pkg.add("DiffEqCallbacks")
# Pkg.add("Plots")
# Pkg.add("BenchmarkTools")
# Pkg.add("KernelAbstractions")
# Pkg.add("Adapt")
# Pkg.add("CUDA")
# Pkg.add("AMDGPU")

using WriteVTK
using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqTsit5
using OrdinaryDiffEqLowStorageRK
using DiffEqCallbacks
using Plots
using BenchmarkTools
using KernelAbstractions
using Adapt
using CUDA
using AMDGPU

include("helper_function.jl")


# Create experiment folder #####################################################
path = joinpath(@__DIR__,"results_gpu")
rm(path, recursive=true, force=true)
mkpath(path)


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

# Earth surface temperature [°C]
ϕs = 20
# Rock temperature as a function of depth [°C]
ϕ0_open(d, ϕs) = ϕs + 0.5 * d
ϕ0(d) = ϕ0_open(d, ϕs) # to make sure we dont work with a global variable

# Rock: granite #########################################
# Rock density [g/m3]
ρr = 2750000
# Rock specific heat [J/(g °C)]
cr = 0.790
# Rock thermal conductivity [W/(m °C)]
λr = 2.62
# Rock diffusion coefficient
dr = λr / (ρr * cr)

# Pipes: polyethylene ####################################
# Inner pipe inside radius [m]
r1 = 0.1
# Inner pipe thickness [m]
t1 = 0.01
# Inner pipe height [m]
h1 = 8.5
# Outer pipe inside radius [m]
vol1 = π * r1^2 * h1
r2 = sqrt((vol1 + π * (r1 + t1)^2 * h1) / (h1 * π)) # <= vol1 = vol2 = π*r2^2*h1-π*(r1+t1)^2*h1
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
dp = λp / (ρp * cp)

# Fluid: water #########################################
# Fluid density [g/m3]
ρf = 997000
# Fluid specific heat capacity [J/(g °C)]
cf = 4.184
# Fluid thermal conductivity [W/(m °C)]
λf = 0.6
# Fluid diffusion coefficient
df = λf / (ρf * cf)
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
Re = ρf * uf * Lf / μf

all_physical_parameters = (; ϕs, ρr, cr, λr, dr, ϕ0, r1, t1, h1, r2, t2, h2, ε,
    ρp, cp, λp, dp, ρf, cf, λf, df, uf, vx0, vy0, vz0, Lf, μf, Re)

# See https://gchem.cm.utexas.edu/data/section2.php?target=heat-capacities.php
#     https://en.wikipedia.org/wiki/High-density_polyethylene
#     https://en.wikipedia.org/wiki/Numerical_solution_of_the_convection%E2%80%93diffusion_equation






function rhs!(dϕ, ϕ, cache, t)
    (; backend, d, vx, vy, vz, gridx, gridy, gridz, Nx, Ny, Nz, xc, yc, r1, t1, r2, ε, dx_inv, dy_inv, dz_inv, ϕ0_val) = cache

    kernel_boundary_x!(backend)(ϕ, Nx, ndrange=(Ny - 2, Nz - 2))
    kernel_boundary_y!(backend)(ϕ, Ny, ndrange=(Nx - 2, Nz - 2))
    kernel_boundary_z!(backend)(ϕ, Nz, ϕ0_val, ndrange=(Nx - 2, Ny - 2))

    # this gives only a slight speedup, but makes it quite a bit more complicated
    # (; N_inner, N_between, idx_map_inner, idx_map_between) = cache
    # kernel_diffusion!(backend)(dϕ, ϕ, d, dx_inv, dy_inv, dz_inv, ndrange=(Nx - 2, Ny - 2, Nz - 2))
    # kernel_convection_inner!(backend)(dϕ, ϕ, vx, vy, vz, ε, dx_inv, dy_inv, dz_inv, idx_map_inner, ndrange=(N_inner, Nz - 2))
    # kernel_convection_between!(backend)(dϕ, ϕ, vx, vy, vz, ε, dx_inv, dy_inv, dz_inv, idx_map_between, ndrange=(N_between, Nz - 2))


    kernel_rhs!(backend)(dϕ, ϕ, d, vx, vy, vz, gridx, gridy, xc, yc, r1, t1, r2, ε, dx_inv, dy_inv, dz_inv, ndrange=(Nx - 2, Ny - 2, Nz - 2))

    return nothing
end

# Simulation ##################################################################


# Geometry distances [m] (include both points 0.0 and 1.0)
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
zmin, zmax = 0.0, h2 + 1

Nx = 101
Ny = 101
Nz = 81

# Center of the pipe
xc = (xmax - xmin) / 2
yc = (ymax - ymin) / 2

gridx = get_grid(Nx, xmin, xmax)
gridy = get_grid(Ny, ymin, ymax)
gridz = get_grid(Nz, zmin, zmax)

# Define temperature matrix
ϕ = zeros(Nx, Ny, Nz)

# Initial and boundary conditions
d = zero(ϕ)
vx = zero(ϕ)
vy = zero(ϕ)
vz = zero(ϕ)

# for big grids, this should also be a Kernel in the future
for (i, x) in enumerate(gridx)
    for (j, y) in enumerate(gridy)
        for (k, z) in enumerate(gridz)
            d[i, j, k], vx[i, j, k], vy[i, j, k], vz[i, j, k], ϕ[i, j, k] =
                initial_condition(x, y, z, xc, yc, all_physical_parameters)
        end
    end
end

V = zeros(3, Nx, Ny, Nz)
V[1, :, :, :] .= vx
V[2, :, :, :] .= vy  
V[3, :, :, :] .= vz


save(path,"velocity",V,gridx,gridy,gridz,0)
save(path,"diff_coeff",d,gridx,gridy,gridz,0)

# change backend depending on hardware
backend = CPU() # or ROCBackend() or CUDABackend() or CPU()

cache = create_cache(backend=backend, d=d, vx=vx, vy=vy, vz=vz, gridx=gridx, gridy=gridy, gridz=gridz, r1=r1, t1=t1, r2=r2, ε=ε)
ϕ_adapt = adapt(backend, ϕ)

tspan = (0.0, 30.0)
prob = ODEProblem(rhs!, ϕ_adapt, tspan, cache)



saveat = range(tspan..., 16)
saveat = 0:2:30 # if it is not a integer, file names will be with decimal point
callback, saved_values = save_and_print_callback(saveat, write_to_file=true)
@time sol = solve(prob, RDPK3SpFSAL35(), save_everystep=false, abstol=1e-3, reltol=1e-3, callback=callback);

for (i, t) in enumerate(saved_values.t)
    temp = saved_values.saveval[i][Nx÷2, Ny÷2, Nz-1]
    println("using adaptive time integration: t = $(round(t, digits = 2)) s, ϕ = $(round(temp, digits = 4)) °C")
end


# to reproduce results from model-current/dbhe-coaxial.jl use Euler method
# (otherwise dont use the Euler method it!!)
saveat = 0:2:30
callback, saved_values_euler = save_and_print_callback(saveat, write_to_file=true, prepend_file="euler_")
@time sol_euler = solve(prob, Euler(), save_everystep=false,
callback=callback,
adaptive=false, dt=1);
# print results to REPL
for (i, t) in enumerate(saved_values_euler.t)
    temp = saved_values_euler.saveval[i][Nx÷2, Ny÷2, Nz-1]
    println("using explicit euler: t = $(t)s, ϕ = $(round(temp, digits = 4))°C")
end

