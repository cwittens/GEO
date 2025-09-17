using Pkg
Pkg.activate(@__DIR__)
# Pkg.instantiate() # if everything works as expected, only run this and not "Pkg.add(...)"

# Pkg.add("Trixi")
# Pkg.add("OrdinaryDiffEqTsit5")
# Pkg.add("OrdinaryDiffEqLowStorageRK")
# Pkg.add("Plots")

using Trixi

using OrdinaryDiffEqTsit5
using OrdinaryDiffEqLowStorageRK
using Plots

include(joinpath(@__DIR__, "diffusion_stuff.jl"))
include(joinpath(@__DIR__, "linear_advection_stuff.jl"))
include(joinpath(@__DIR__, "helper_functions.jl"))

using Base.Threads
println("Julia has access to $(Threads.nthreads()) threads")

################################################################################
# Diffusion convection equation for temperature in a pipe-in-pipe geometry
#
# Equation: ρ c ∂ϕ/∂t =  ∂(λ ∂ϕ/∂x)/∂x + ∂(λ ∂ϕ/∂y)/∂y + ∂(λ ∂ϕ/∂z)/∂z
#                        -(ε vx ∂ϕ/∂x + ε vy ∂ϕ/∂y + ε vz ∂ϕ/∂z)
#                        + source
# OR:
# ∂ϕ/∂t + ∂(ε'vx ϕ - d ∂ϕ/∂x)/∂x + ∂(ε'vy ϕ - d ∂ϕ/∂y)/∂y + ∂(ε'vz ϕ - d ∂ϕ/∂z)/∂z
# = 0


all_physical_parameters = set_up_physics()

function initial_condition_borehole_params(x, t, equation, all_physical_parameters)
    (; r1, t1, r2, t2, ϕ0, df, uf, ϕs, dp, xc, yc, dr) = all_physical_parameters
    r = sqrt((x[1] - xc)^2 + (x[2] - yc)^2)
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
        ϕ_val = ϕ0(x[3])
    end

    SVector(ϕ_val, vx_val, vy_val, vz_val, d_val)
end

initial_condition_borehole(x, t, equation) = initial_condition_borehole_params(x, t, equation, all_physical_parameters)

initial_condition = initial_condition_borehole



########################################################
# test setup for advection diffusion equation
########################################################
equations_hyperbolic = DiffusionConvectionHyperbolic3D()
equations_parabolic = DiffusionConvectionParabolic3D(equations_hyperbolic)



zend = 0
T_zend = all_physical_parameters.ϕ0(zend)

# domain needs to be a cube for TreeMesh
# TODO: change to non-dimensional coordinates to account for different lengths in x,y,z direction
coordinates_min = (0.0, 0.0, 0.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = (10.0, 10.0, 10.0) # maximum coordinates (max(x), max(y), max(z))


region_min0 = (0.0, 0.0, 0.0)
region_max0 = (2.0, 2.0, 10.0)

region_min1 = (0.0, 0.0, 0.0)
region_max1 = (1.0, 1.0, 10.0)


region_min2 = (0.2, 0.2, 0.0)
region_max2 = (0.8, 0.8, 10.0)

refinement_patches = (
    # First refinement patch 
    (type="box",
        coordinates_min=region_min0,
        coordinates_max=region_max0),

    # Second refinement patch - even finer 
    (type="box",
        coordinates_min=region_min1,
        coordinates_max=region_max1),

        (type="box",
        coordinates_min=region_min1,
        coordinates_max=region_max1),

    # Third refinement patch - even finer
    (type="box",
        coordinates_min=region_min2,
        coordinates_max=region_max2),
    #
    # (type="box",
    #     coordinates_min=region_min2,
    #     coordinates_max=region_max2),

    #
    (type="box",
        coordinates_min=region_min2,
        coordinates_max=region_max2),)

mesh = TreeMesh(coordinates_min, coordinates_max,
    initial_refinement_level=3,
    refinement_patches=refinement_patches,
    n_cells_max=1000000,
    periodicity=false)



# Create a DGSEM solver with polynomials of degree `polydeg`
# Remember to pass a tuple of the form `(conservative_flux, nonconservative_flux)`
# as `surface_flux` and `volume_flux` when working with nonconservative terms
volume_flux = (flux_central, flux_nonconservative)
surface_flux = (flux_lax_friedrichs, flux_nonconservative)
solver = DGSEM(polydeg=3, surface_flux=surface_flux,
    volume_integral=VolumeIntegralFluxDifferencing(volume_flux))



boundary_conditions_hyperbolic = (;
    x_neg=boundary_condition_do_nothing,
    y_neg=boundary_condition_do_nothing,
    z_neg=boundary_condition_do_nothing,
    y_pos=boundary_condition_do_nothing,
    x_pos=boundary_condition_do_nothing,
    z_pos=BoundaryConditionDirichlet(initial_condition)
)

bc_neumann = BoundaryConditionNeumann((x, t, equations) -> SVector(0.0))
boundary_conditions_parabolic = (;
    x_neg=bc_neumann,
    y_neg=bc_neumann,
    z_neg=bc_neumann,
    y_pos=bc_neumann,
    x_pos=bc_neumann,
    z_pos=BoundaryConditionDirichlet(initial_condition)
)



semi = SemidiscretizationHyperbolicParabolic(mesh,
    (equations_hyperbolic, equations_parabolic),
    initial_condition,
    solver;
    solver_parabolic=ViscousFormulationBassiRebay1(),
    boundary_conditions=(boundary_conditions_hyperbolic, boundary_conditions_parabolic))
# Create a dummy solution (at t=0)
ode = semidiscretize(semi, (0.0, 0.0))
u0 = ode.u0

# Create plot data and visualize the mesh
pd = PlotData2D(u0, semi)
plot(getmesh(pd), xlims=(0.0, 1.0), ylims=(0.0, 1.0))


tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)
callbacks = CallbackSet(SummaryCallback(), AliveCallback(analysis_interval=10))
time_int_tol = 1.0e-3

saveat = range(tspan..., 11)
sol = solve(ode, RDPK3SpFSAL49(); abstol=time_int_tol, reltol=time_int_tol,
saveat=saveat,
    callback=callbacks);

begin
    pd = PlotData2D(sol.u[end], semi, slice=:xy, point=(0.0, 0.0, 9))
    p = plot(pd["ε*vz"],)# clims=(20.0, 25.0))
    # plot!(getmesh(pd))
    plot!(p, xlims=(0.0, 1.0), ylims=(0.0, 1.0))
end

