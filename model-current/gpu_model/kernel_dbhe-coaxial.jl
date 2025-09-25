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

include(joinpath(@__DIR__, "helper_function_kernel.jl"))


# Create experiment folder #####################################################
path = joinpath(@__DIR__, "results_gpu")
# rm(path, recursive=true, force=true)
# mkpath(path)


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

all_physical_parameters = set_up_physics()


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

nn = 1
Nx = nn * 100 + 1
Ny = nn * 100 + 1
Nz = nn * 80 + 1

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
                initial_condition2(x, y, z, xc, yc, all_physical_parameters)
        end
    end
end

V = zeros(3, Nx, Ny, Nz)
V[1, :, :, :] .= vx
V[2, :, :, :] .= vy
V[3, :, :, :] .= vz


# save(path, "velocity", V, gridx, gridy, gridz, 0)
# save(path, "diff_coeff", d, gridx, gridy, gridz, 0)

# change backend depending on hardware
backend = CPU() # or ROCBackend() or CUDABackend() or CPU()

cache = create_cache(backend=backend, d=d, vx=vx, vy=vy, vz=vz, gridx=gridx, gridy=gridy, gridz=gridz, r1=r1, t1=t1, r2=r2, ε=ε)
ϕ_adapt = adapt(backend, ϕ)

tspan = (0.0, 2400.0)
prob = ODEProblem(rhs!, ϕ_adapt, tspan, cache)



saveat = range(tspan..., 160)
# saveat = 0:2:30 # if it is not a integer, file names will be with decimal point
callback, saved_values = save_and_print_callback(saveat, write_to_file=false)
@time sol = solve(prob, Tsit5(), save_everystep=false, abstol=1e-3, reltol=1e-3, callback=callback);

for (i, t) in enumerate(saved_values.t)
    temp = saved_values.saveval[i][Nx÷2, Ny÷2, Nz-1]
    println("using adaptive time integration: t = $(round(t, digits = 2)) s, ϕ = $(round(temp, digits = 4)) °C")
end


begin
    i = 150
    zslice = gridz[Nz-1]
    heatmap(saved_values.saveval[i][:, :, Nz-1]',
        # clims=(20.0, 25.0),
        aspect_ratio=1,
        title="t = $(round(saved_values.t[i], digits=2)) s, z = $(round(zslice, digits=3)) m",
        xlabel="x [m]", ylabel="y [m]",
        size=(600, 500))
end


# to reproduce results from model-current/dbhe-coaxial.jl use Euler method
# (otherwise dont use the Euler method it!!)
saveat = 0:2:30
callback, saved_values_euler = save_and_print_callback(saveat, write_to_file=true, prepend_file="euler_")
@time sol_euler = solve(prob, Euler(), save_everystep=false, dt=1, callback=callback);

# print results to REPL
for (i, t) in enumerate(saved_values_euler.t)
    temp = saved_values_euler.saveval[i][Nx÷2, Ny÷2, Nz-1]
    println("using explicit euler: t = $(t)s, ϕ = $(round(temp, digits = 4))°C")
end

# using adaptive time integration: t = 0.0 s, ϕ = 20.0 °C
# using adaptive time integration: t = 2.0 s, ϕ = 21.9061 °C
# using adaptive time integration: t = 4.0 s, ϕ = 23.0856 °C
# using adaptive time integration: t = 6.0 s, ϕ = 23.8155 °C
# using adaptive time integration: t = 8.0 s, ϕ = 24.2669 °C
# using adaptive time integration: t = 10.0 s, ϕ = 24.547 °C
# using adaptive time integration: t = 12.0 s, ϕ = 24.7191 °C
# using adaptive time integration: t = 14.0 s, ϕ = 24.8268 °C
# using adaptive time integration: t = 16.0 s, ϕ = 24.8932 °C
# using adaptive time integration: t = 18.0 s, ϕ = 24.9329 °C
# using adaptive time integration: t = 20.0 s, ϕ = 24.9592 °C
# using adaptive time integration: t = 22.0 s, ϕ = 24.9757 °C
# using adaptive time integration: t = 24.0 s, ϕ = 24.9844 °C
# using adaptive time integration: t = 26.0 s, ϕ = 24.9892 °C
# using adaptive time integration: t = 28.0 s, ϕ = 24.9933 °C
# using adaptive time integration: t = 30.0 s, ϕ = 24.9958 °C

# nn = 4 with z = gridz[Nz-1] = 9.975
# using adaptive time integration: t = 0.0 s, ϕ = 20.0 °C
# using adaptive time integration: t = 2.0 s, ϕ = 22.3636 °C
# using adaptive time integration: t = 4.0 s, ϕ = 23.6098 °C
# using adaptive time integration: t = 6.0 s, ϕ = 24.2669 °C
# using adaptive time integration: t = 8.0 s, ϕ = 24.6141 °C
# using adaptive time integration: t = 10.0 s, ϕ = 24.796 °C
# using adaptive time integration: t = 12.0 s, ϕ = 24.8935 °C
# using adaptive time integration: t = 14.0 s, ϕ = 24.9426 °C
# using adaptive time integration: t = 16.0 s, ϕ = 24.9709 °C
# using adaptive time integration: t = 18.0 s, ϕ = 24.985 °C
# using adaptive time integration: t = 20.0 s, ϕ = 24.9906 °C
# using adaptive time integration: t = 22.0 s, ϕ = 24.9956 °C
# using adaptive time integration: t = 24.0 s, ϕ = 24.9983 °C
# using adaptive time integration: t = 26.0 s, ϕ = 24.9986 °C
# using adaptive time integration: t = 28.0 s, ϕ = 24.9986 °C
# using adaptive time integration: t = 30.0 s, ϕ = 24.9992 °C