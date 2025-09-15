using Pkg
Pkg.activate(@__DIR__)
# Pkg.instantiate() # if everything works as expected, only run this and not "Pkg.add(...)"

# Pkg.add("Trixi")
# Pkg.add("OrdinaryDiffEqTsit5")
# Pkg.add("OrdinaryDiffEqLowStorageRK")
# Pkg.add("Plots")

using Trixi
using Trixi: AbstractEquations, get_node_vars, AbstractEquationsParabolic, @threaded
import Trixi: varnames, default_analysis_integrals, flux, max_abs_speed_naive,
    have_nonconservative_terms

using OrdinaryDiffEqTsit5
using OrdinaryDiffEqLowStorageRK
using Plots

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


begin # define a new equation type for the nonconservative linear advection equation
#! format: noindent

# Since there is no native support for variable coefficients, we use more
# variables: one for the basic unknowns `ϕ` and another three for the coefficient `ε*v's`
# and one for the diffusion coefficient `d = λ/(ρ c)`  with 3 dimensions, 5 variables (ϕ, εvx, εvy, εvz, d)
struct DiffusionConvectionHyperbolic3D <: AbstractEquations{3,5} end


function varnames(::typeof(cons2cons), ::DiffusionConvectionHyperbolic3D)
    ("phi", "ε*vx", "ε*vy", "ε*vz", "d")
end


default_analysis_integrals(::DiffusionConvectionHyperbolic3D) = ()


# The conservative part of the flux is zero
flux(u, orientation, equation::DiffusionConvectionHyperbolic3D) = zero(u)

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
    ::DiffusionConvectionHyperbolic3D)
    # Extract velocity components from left and right states
    _, vx_ll, vy_ll, vz_ll, _ = u_ll
    _, vx_rr, vy_rr, vz_rr, _ = u_rr

    # Select velocity component based on orientation
    if orientation == 1
        v_ll = vx_ll
        v_rr = vx_rr
    elseif orientation == 2
        v_ll = vy_ll
        v_rr = vy_rr
    else  # orientation == 3
        v_ll = vz_ll
        v_rr = vz_rr
    end

    return max(abs(v_ll), abs(v_rr))
end


# We use nonconservative terms
have_nonconservative_terms(::DiffusionConvectionHyperbolic3D) = Trixi.True()


# This "nonconservative numerical flux" implements the nonconservative terms for 3D
# The nonconservative term is: (ε vx ∂ϕ/∂x + ε vy ∂ϕ/∂y + ε vz ∂ϕ/∂z)
# In general, nonconservative terms can be written in the form
#   g(u) ∂ₓ h(u)
# Thus, a discrete difference approximation of this nonconservative term needs
# - `u mine`:  the value of `u` at the current position (for g(u))
# - `u_other`: the values of `u` in a neighborhood of the current position (for ∂ₓ h(u))
function flux_nonconservative(u_mine, u_other, orientation,
    equations::DiffusionConvectionHyperbolic3D)
    # Extract variables from u_mine (gives velocity components at current position)
    # TODO: check if unpacking in the if else is faster
    _, vx_mine, vy_mine, vz_mine, _ = u_mine

    # Extract variables from u_other (gives phi value at neighboring position for gradient)
    phi_other, _, _, _, _ = u_other

    # Select the appropriate velocity component based on spatial orientation
    if orientation == 1        # x-direction
        v_component = vx_mine
    elseif orientation == 2    # y-direction  
        v_component = vy_mine
    else  # orientation == 3   # z-direction
        v_component = vz_mine
    end

    # Return contributions to each equation
    return SVector(v_component * phi_other,  # contribution to ϕ equation
        zero(phi_other),           # no contribution to vx equation (auxiliary)
        zero(phi_other),           # no contribution to vy equation (auxiliary)
        zero(phi_other),           # no contribution to vz equation (auxiliary)
        zero(phi_other))           # no contribution to d equation (auxiliary)
end
end #end begin block

# begin # define a new equation type for the diffusion equation
#! format: noindent


# Since there is no native support for variable coefficients, we use more
# variables: one for the basic unknowns `ϕ` and another three for the coefficient `ε*v's`
# and one for the diffusion coefficient `d = λ/(ρ c)`
struct DiffusionConvectionParabolic3D{E} <: AbstractEquationsParabolic{3,5,GradientVariablesConservative}
    equations_hyperbolic::E
end

function varnames(variable_mapping, equations_parabolic::DiffusionConvectionParabolic3D)
    varnames(variable_mapping, equations_parabolic.equations_hyperbolic)
end


function flux(u, gradients, orientation::Integer,
    equations_parabolic::DiffusionConvectionParabolic3D)
    _, _, _, _, diffusivity = u
    null = zero(diffusivity)

    dudx, dudy, dudz = gradients
    if orientation == 1
        return SVector(diffusivity * dudx[1], null, null, null, null)
    elseif orientation == 2
        return SVector(diffusivity * dudy[1], null, null, null, null)
    else
        return SVector(diffusivity * dudz[1], null, null, null, null)
    end
end

# TODO: this is most likely wrong. need to think what do to here
# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
    ::Union{DiffusionConvectionHyperbolic3D, DiffusionConvectionParabolic3D})
    # Extract velocity components from left and right states
    _, vx_ll, vy_ll, vz_ll, _ = u_ll
    _, vx_rr, vy_rr, vz_rr, _ = u_rr

    # Project velocities onto normal direction
    v_normal_ll = vx_ll * normal_direction[1] + vy_ll * normal_direction[2] + vz_ll * normal_direction[3]
    v_normal_rr = vx_rr * normal_direction[1] + vy_rr * normal_direction[2] + vz_rr * normal_direction[3]

    return max(abs(v_normal_ll), abs(v_normal_rr))
end




########################################################
# test setup for advection diffusion equation
########################################################
equations_hyperbolic = DiffusionConvectionHyperbolic3D()
equations_parabolic = DiffusionConvectionParabolic3D(equations_hyperbolic)



function initial_condition_test_open2(x, t, equation, all_physical_parameters)
    (; ε) = all_physical_parameters
    x_trans = x
    vx = 1.2
    vy = 0.7
    vz = 3.5
    nu = 5.0e-2
    c = 1.0
    A = 0.5
    L = 2
    f = 1 / L
    omega = 2 * pi * f
    phi = c + A * sin(omega * sum(x_trans)) * exp(-2 * nu * omega^2 * t)

    SVector(phi, vx, vy, vz, nu)
end

initial_condition_test2(x, t, equations) = initial_condition_test_open2(x, t, equations, all_physical_parameters)

function initial_condition_borehole_params(x, t, equation, all_physical_parameters)
    (; r1, t1, r2, t2, ϕ0, df, uf, ϕs, dp, xc, yc) = all_physical_parameters
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
        ϕ_val = ϕ0(z)
    end

    SVector(ϕ_val, vx_val, vy_val, vz_val, d_val)
end

initial_condition_borehole(x, t, equation) = initial_condition_borehole_params(x, t, equation, all_physical_parameters)

# define periodic boundary conditions everywhere
boundary_conditions = boundary_condition_periodic
boundary_conditions_parabolic = boundary_condition_periodic

# Create a DGSEM solver with polynomials of degree `polydeg`
# Remember to pass a tuple of the form `(conservative_flux, nonconservative_flux)`
# as `surface_flux` and `volume_flux` when working with nonconservative terms
volume_flux = (flux_central, flux_nonconservative)
surface_flux = (flux_lax_friedrichs, flux_nonconservative)

solver = DGSEM(polydeg=3, surface_flux=surface_flux,
    volume_integral=VolumeIntegralFluxDifferencing(volume_flux))


coordinates_min = (0.0, 0.0, 0.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = (1.0, 1.0, h2 + 1.0) # maximum coordinates (max(x), max(y), max(z))

    coordinates_min = (-1.0, -1.0, -1.0)
    coordinates_max = (1.0, 1.0, h2 + 1.0)  # h2 = 9, so (1.0, 1.0, 10.0)
    coordinates_max = (1.0, 1.0,  1.0) 
mesh = TreeMesh(coordinates_min, coordinates_max,
    initial_refinement_level=3,
    n_cells_max=80_000)

function create_geothermal_p4est_mesh(all_physical_parameters)
    (; h2) = all_physical_parameters
    
    # Domain to match your finite difference setup
    coordinates_min = (-1.0, -1.0, -1.0)
    coordinates_max = (1.0, 1.0, h2 + 1.0)  # h2 = 9, so (1.0, 1.0, 10.0)
    coordinates_max = (1.0, 1.0,  1.0) 

    trees_per_dimension = (4, 4, 4)
    
    mesh = P4estMesh(trees_per_dimension,
                     polydeg = 3,
                     coordinates_min = coordinates_min,
                     coordinates_max = coordinates_max,
                     initial_refinement_level = 0,
                     periodicity = (true, true, true))
    
    return mesh
end  

mesh = create_geothermal_p4est_mesh(all_physical_parameters)

semi = SemidiscretizationHyperbolicParabolic(mesh, 
(equations_hyperbolic, equations_parabolic), initial_condition_test2, solver;
    # solver_parabolic=ViscousFormulationBassiRebay1(),
    boundary_conditions=(boundary_conditions, boundary_conditions_parabolic))

tspan = (0.0, 1.5)
ode = semidiscretize(semi, tspan)
callbacks = CallbackSet(SummaryCallback(), AliveCallback(analysis_interval = 100))
time_int_tol = 1.0e-4

sol = solve(ode, Tsit5(); abstol=time_int_tol, reltol=time_int_tol,
    ode_default_options()..., callback=callbacks);

pd = PlotData1D(sol, semi)
plot(pd["phi"])
plot!(getmesh(pd))

sol


begin
    # Create a simple simulation setup to check everything works so far


    equation = NonconservativeLinearAdvectionEquation3D()

    function initial_condition_test_open(x, t, equation::NonconservativeLinearAdvectionEquation3D, all_physical_parameters)
        #@show x
        (; ε) = all_physical_parameters
        phi = sin(x[2])
        advection_velocity_x = ε
        SVector(phi, 0.0, advection_velocity_x, 0.0, 0.0)  # Assuming advection in x-direction only
    end


    initial_condition_test(x, t, equation::NonconservativeLinearAdvectionEquation3D) = initial_condition_test_open(x, t, equation::NonconservativeLinearAdvectionEquation3D, all_physical_parameters)


    coordinates_min = (0.0, 0.0, 0.0)
    coordinates_max = (2π, 2π, 2π)

    mesh = TreeMesh(coordinates_min, coordinates_max,
        initial_refinement_level=3,
        n_cells_max=30_000)


    # Create a DGSEM solver with polynomials of degree `polydeg`
    # Remember to pass a tuple of the form `(conservative_flux, nonconservative_flux)`
    # as `surface_flux` and `volume_flux` when working with nonconservative terms
    volume_flux = (flux_central, flux_nonconservative)
    surface_flux = (flux_lax_friedrichs, flux_nonconservative)
    solver = DGSEM(polydeg=3, surface_flux=surface_flux,
        volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

    # Setup the spatial semidiscretization containing all ingredients
    semi = SemidiscretizationHyperbolic(mesh, equation, initial_condition_test, solver)

    # Create an ODE problem with given time span
    tspan = (0.0, 2.1)
    ode = semidiscretize(semi, tspan)



    # Set up some standard callbacks summarizing the simulation setup and computing
    # errors of the numerical solution
    summary_callback = SummaryCallback()
    analysis_callback = AnalysisCallback(semi, interval=50)
    callbacks = CallbackSet(summary_callback, analysis_callback)

    # OrdinaryDiffEq's `solve` method evolves the solution in time and executes
    # the passed callbacks
    sol = solve(ode, Tsit5(), abstol=1.0e-6, reltol=1.0e-6;
        ode_default_options()..., callback=callbacks)


    # Plot the numerical solution at the final time
    pd = PlotData2D(sol)#,  slice=:y)
    plot(pd["phi"])
    plot!(getmesh(pd))

    pd = PlotData1D(sol, slice=:y)
    plot(pd["phi"])
    plot!(getmesh(pd))

end





