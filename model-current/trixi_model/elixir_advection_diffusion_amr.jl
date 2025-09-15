using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (1.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

diffusivity() = 5.0e-2
equations_parabolic = LaplaceDiffusion3D(diffusivity(), equations)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 80_000)

# Define initial condition
function initial_condition_diffusive_convergence_test(x, t,
                                                      equation::LinearScalarAdvectionEquation3D)
    # Store translated coordinate for easy use of exact solution
    x_trans = x - equation.advection_velocity * t

    nu = diffusivity()
    c = 1.0
    A = 0.5
    L = 2
    f = 1 / L
    omega = 2 * pi * f
    scalar = c + A * sin(omega * sum(x_trans)) * exp(-2 * nu * omega^2 * t)
    return SVector(scalar)
end
initial_condition = initial_condition_diffusive_convergence_test

# define periodic boundary conditions everywhere
boundary_conditions = boundary_condition_periodic
boundary_conditions_parabolic = boundary_condition_periodic

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolicParabolic(mesh,
                                             (equations, equations_parabolic),
                                             initial_condition, solver;
                                             solver_parabolic = ViscousFormulationBassiRebay1(),
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100


alive_callback = AliveCallback(analysis_interval = analysis_interval)



callbacks = CallbackSet(summary_callback,
                        alive_callback,
 )

###############################################################################
# run the simulation

sol_elixir = solve(ode, Tsit5(); abstol = 1e-6, reltol = 1e-6,
            ode_default_options()..., callback = callbacks);

pd_elixir = PlotData2D(sol_elixir)
plot(pd_elixir)
plot!(getmesh(pd_elixir))
