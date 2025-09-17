#  reproduce Trixi.jl/examples/tree_3d_dgsem/elixir_advection_diffusion_nonperiodic.jl

using Pkg
Pkg.activate(@__DIR__)

using Trixi
using OrdinaryDiffEqLowStorageRK
using Plots

include(joinpath(@__DIR__, "diffusion_stuff.jl"))
include(joinpath(@__DIR__, "linear_advection_stuff.jl"))


diffusivity() = 5.0e-2
function initial_condition_eriksson_johnson(x, t, equations)
    vx = 1.0
    vy = 0.0
    vz = 0.0
    d = diffusivity()
    l = 4
    epsilon = d
    lambda_1 = (-1 + sqrt(1 - 4 * epsilon * l)) / (-2 * epsilon)
    lambda_2 = (-1 - sqrt(1 - 4 * epsilon * l)) / (-2 * epsilon)
    r1 = (1 + sqrt(1 + 4 * pi^2 * epsilon^2)) / (2 * epsilon)
    s1 = (1 - sqrt(1 + 4 * pi^2 * epsilon^2)) / (2 * epsilon)
    u = exp(-l * t) * (exp(lambda_1 * x[1]) - exp(lambda_2 * x[1])) +
        cos(pi * x[2]) * (exp(s1 * x[1]) - exp(r1 * x[1])) / (exp(-s1) - exp(-r1))
    return SVector(u, vx, vy, vz, d)
end
initial_condition = initial_condition_eriksson_johnson

equations_hyperbolic = DiffusionConvectionHyperbolic3D()
equations_parabolic = DiffusionConvectionParabolic3D(equations_hyperbolic)


volume_flux = (flux_central, flux_nonconservative)
surface_flux = (flux_lax_friedrichs, flux_nonconservative)
solver = DGSEM(polydeg=3, surface_flux=surface_flux,
    volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -0.5, -0.5) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = (0.0, 0.5, 0.5) # maximum coordinates (max(x), max(y), max(z))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
    initial_refinement_level=3,
    periodicity=false,
    n_cells_max=80_000) # set maximum capacity of tree data structure

boundary_conditions_hyperbolic = (;
    x_neg=boundary_condition_do_nothing,
    y_neg=BoundaryConditionDirichlet(initial_condition),
    z_neg=boundary_condition_do_nothing,
    y_pos=BoundaryConditionDirichlet(initial_condition),
    x_pos=boundary_condition_do_nothing,
    z_pos=boundary_condition_do_nothing)

boundary_conditions_parabolic = BoundaryConditionDirichlet(initial_condition)

semi = SemidiscretizationHyperbolicParabolic(mesh,
                                             (equations_hyperbolic, equations_parabolic),
                                             initial_condition, solver;
                                             solver_parabolic = ViscousFormulationBassiRebay1(),
                                             boundary_conditions = (boundary_conditions_hyperbolic,
                                                                    boundary_conditions_parabolic))
# Create ODE problem with time span `tspan`
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

callbacks = CallbackSet(SummaryCallback(), AliveCallback(analysis_interval = 100))

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
time_int_tol = 1.0e-7
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
begin
    pd = PlotData2D(sol)
    plot(pd["phi"], clims=(0.0, 1.0))
    plot!(getmesh(pd))
end
