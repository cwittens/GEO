using Pkg
Pkg.activate(@__DIR__)
# Pkg.instantiate() # if everything works as expected, only run this and not "Pkg.add(...)"

# add more packages if needed here
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


# Simulation parameters #########################################################

# Simulation time [s]
tspan = (0.0, 300.0)

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


function get_grid(N, min, max)
    return range(min, max, length=N)
end

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





function initial_condition(x, y, z, xc, yc, all_physical_parameters)
    (; r1, t1, r2, t2, ϕ0, df, uf, ϕs, dp) = all_physical_parameters
    r = sqrt((x - xc)^2 + (y - yc)^2)
    if r < r1 # inside inner pipe
        d_val = df
        vx_val = 0
        vy_val = 0
        vz_val = -uf
        ϕ_val = ϕs
    elseif r < r1 + t1 # inner pipe
        d_val = dp
        vx_val = 0
        vy_val = 0
        vz_val = 0
        ϕ_val = ϕs
    elseif r < r2 # between inner and outer pipe
        d_val = df
        vx_val = 0
        vy_val = 0
        vz_val = uf
        ϕ_val = ϕs
    elseif r < r2 + t2 # outer pipe
        d_val = dp
        vx_val = 0
        vy_val = 0
        vz_val = 0
        ϕ_val = ϕs
    else # rock
        d_val = dr
        vx_val = 0
        vy_val = 0
        vz_val = 0
        ϕ_val = ϕ0(z)
    end

    return d_val, vx_val, vy_val, vz_val, ϕ_val

end

for (i, x) in enumerate(gridx)
    for (j, y) in enumerate(gridy)
        for (k, z) in enumerate(gridz)
            d[i, j, k], vx[i, j, k], vy[i, j, k], vz[i, j, k], ϕ[i, j, k] =
                initial_condition(x, y, z, xc, yc, all_physical_parameters)
        end
    end
end

# TODO first only one kernel. Later diff and conv in separate kernels with pre mapped inside pipe and between pipes coordinates
@kernel function kernel_rhs!(dϕ, @Const(ϕ), @Const(d), @Const(vx), @Const(vy), @Const(vz), @Const(gridx), @Const(gridy), xc, yc, r1, t1, r2, ε, dx_inv, dy_inv, dz_inv)
    i, j, k = @index(Global, NTuple)
    i, j, k = i + 1, j + 1, k + 1

    half = eltype(ϕ)(0.5)

    # Convective term
    x = gridx[i]
    y = gridy[j]
     r = sqrt((x - xc)^2 + (y - yc)^2)
    if r < r1 # inside inner pipe
        conv = (
            ε * vx[i, j, k] * (ϕ[i+1, j, k] - ϕ[i, j, k]) * dx_inv
            + ε * vy[i, j, k] * (ϕ[i, j+1, k] - ϕ[i, j, k]) * dy_inv
            + ε * vz[i, j, k] * (ϕ[i, j, k+1] - ϕ[i, j, k]) * dz_inv
        )
    elseif r1 + t1 < r < r2 # between inner and outer pipe 
        conv = (
            ε * vx[i, j, k] * (ϕ[i, j, k] - ϕ[i-1, j, k]) * dx_inv
            + ε * vy[i, j, k] * (ϕ[i, j, k] - ϕ[i, j-1, k]) * dy_inv
            + ε * vz[i, j, k] * (ϕ[i, j, k] - ϕ[i, j, k-1]) * dz_inv
        )
    else
        conv = 0
    end

    # Diffusive term
    diff = (
        (d[i+1, j, k] + d[i, j, k]) * half * (ϕ[i+1, j, k] - ϕ[i, j, k]) * dx_inv
        -
        (d[i, j, k] + d[i-1, j, k]) * half * (ϕ[i, j, k] - ϕ[i-1, j, k]) * dx_inv
    ) * dx_inv
    +
    (
        (d[i, j+1, k] + d[i, j, k]) * half * (ϕ[i, j+1, k] - ϕ[i, j, k]) * dy_inv
        -
        (d[i, j, k] + d[i, j-1, k]) * half * (ϕ[i, j, k] - ϕ[i, j-1, k]) * dy_inv
    ) * dy_inv
    +
    (
        (d[i, j, k+1] + d[i, j, k]) * half * (ϕ[i, j, k+1] - ϕ[i, j, k]) * dz_inv
        -
        (d[i, j, k] + d[i, j, k-1]) * half * (ϕ[i, j, k] - ϕ[i, j, k-1]) * dz_inv
    ) * dz_inv


    dϕ[i, j, k] = diff - conv

end

@kernel function kernel_boundary_x!(ϕ, Nx)
    j, k = @index(Global, NTuple)
    j, k = j + 1, k + 1

    # Do both x boundaries in one kernel
    ϕ[1, j, k] = ϕ[2, j, k]        # left boundary
    ϕ[Nx, j, k] = ϕ[Nx-1, j, k]    # right boundary
end

@kernel function kernel_boundary_y!(ϕ, Ny)
    i, k = @index(Global, NTuple)
    i, k = i + 1, k + 1

    # Do both y boundaries in one kernel
    ϕ[i, 1, k] = ϕ[i, 2, k]       # front boundary  
    ϕ[i, Ny, k] = ϕ[i, Ny-1, k]   # back boundary
end

@kernel function kernel_boundary_z!(ϕ, Nz, ϕ0_val)
    i, j = @index(Global, NTuple)
    i, j = i + 1, j + 1

    # Do both z boundaries in one kernel
    ϕ[i, j, 1] = ϕ[i, j, 2]       # top boundary
    ϕ[i, j, Nz] = ϕ0_val          # bottom boundary (Dirichlet)
end




function rhs!(dϕ, ϕ, cache, t)
    (; backend, d, vx, vy, vz, gridx, gridy, gridz, Nx, Ny, Nz, xc, yc, r1, t1, r2, ε, dx_inv, dy_inv, dz_inv, ϕ0_val) = cache

    kernel_boundary_x!(backend)(ϕ, Nx, ndrange=(Ny - 2, Nz - 2))
    kernel_boundary_y!(backend)(ϕ, Ny, ndrange=(Nx - 2, Nz - 2))
    kernel_boundary_z!(backend)(ϕ, Nz, ϕ0_val, ndrange=(Nx - 2, Ny - 2))


    kernel_rhs!(backend)(dϕ, ϕ, d, vx, vy, vz, gridx, gridy, xc, yc, r1, t1, r2, ε, dx_inv, dy_inv, dz_inv, ndrange=(Nx - 2, Ny - 2, Nz - 2))

    return nothing
end


backend = ROCBackend() # or ROCBackend() or CUDABackend() or CPU()
# backend = CPU()
cache = create_cache(backend=backend, d=d, vx=vx, vy=vy, vz=vz, gridx=gridx, gridy=gridy, gridz=gridz, r1=r1, t1=t1, r2=r2, ε=ε)
ϕ_adapt = adapt(backend, ϕ)

tspan = (0.0, 2400.0)
prob = ODEProblem(rhs!, ϕ_adapt, tspan, cache)



saveat = range(tspan..., 100)
@time sol = solve(prob, RDPK3SpFSAL35(), save_everystep=false, abstol=1e-3, reltol=1e-3);#, saveat=saveat);

temp = [adapt(CPU(),sol[i])[50, 50, 80] for i in 1:length(sol.t)]

for (i, t) in enumerate(sol.t)
    println("using adaptive time integration: t = $(round(t, digits = 2)) s, ϕ = $(round(temp[i], digits = 4)) °C")
end


# to reproduce results from model-current/dbhe-coaxial.jl use Euler method
# (otherwise dont use it!!)
saveat = 0:2:30
@time sol_euler = solve(prob, Euler(), save_everystep=false, saveat=saveat, adaptive=false, dt=1.0);
temp_euler = [adapt(CPU(),sol_euler[i])[50, 50, 80] for i in 1:length(sol_euler.t)]

for (i, t) in enumerate(saveat)
    println("using explicit euler: t = $(t)s, ϕ = $(round(temp_euler[i], digits = 4))°C")
end

