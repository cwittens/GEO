using Pkg
Pkg.activate(@__DIR__)
# Pkg.instantiate() # if everything works as expected, only run this and not "Pkg.add(...)"

Pkg.add("Trixi")
Pkg.add("OrdinaryDiffEqTsit5")
Pkg.add("Plots")

using Trixi
using Trixi: AbstractEquations, get_node_vars
import Trixi: varnames, default_analysis_integrals, flux, max_abs_speed_naive,
              have_nonconservative_terms
using OrdinaryDiffEqTsit5
using Plots

include(joinpath(@__DIR__, "helper_functions.jl"))


################################################################################
# Diffusion convection equation for temperature in a pipe-in-pipe geometry
#
# Equation: ρ c ∂ϕ/∂t =  ∂(λ ∂ϕ/∂x)/∂x + ∂(λ ∂ϕ/∂y)/∂y + ∂(λ ∂ϕ/∂z)/∂z
#                        -(ε vx ∂ϕ/∂x + ε vy ∂ϕ/∂y + ε vz ∂ϕ/∂z)
#                        + source


all_physical_parameters = set_up_physic()


# Since there is no native support for variable coefficients, we use more
# variables: one for the basic unknowns `ϕ` and another three for the coefficient `ε*v's`
struct NonconservativeLinearAdvectionEquation3D <: AbstractEquations{3, # spatial dimension
                                                                   4} # four variables (ϕ, εvx, εvy, εvz)
end

function varnames(::typeof(cons2cons), ::NonconservativeLinearAdvectionEquation3D)
    ("phi", "ε*vx", "ε*vy", "ε*vz")
end


default_analysis_integrals(::NonconservativeLinearAdvectionEquation3D) = ()


# The conservative part of the flux is zero
flux(u, orientation, equation::NonconservativeLinearAdvectionEquation3D) = zero(u)

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                             ::NonconservativeLinearAdvectionEquation3D)
    # Extract velocity components from left and right states
    _, vx_ll, vy_ll, vz_ll = u_ll
    _, vx_rr, vy_rr, vz_rr = u_rr
    
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
have_nonconservative_terms(::NonconservativeLinearAdvectionEquation3D) = Trixi.True()


# This "nonconservative numerical flux" implements the nonconservative terms for 3D
# The nonconservative term is: (ε vx ∂ϕ/∂x + ε vy ∂ϕ/∂y + ε vz ∂ϕ/∂z)
function flux_nonconservative(u_mine, u_other, orientation,
                              equations::NonconservativeLinearAdvectionEquation3D)
    # Extract variables from u_mine (gives velocity components at current position)
    # TODO: check if unpacking in the if else is faster
    _, vx_mine, vy_mine, vz_mine = u_mine
    
    # Extract variables from u_other (gives phi value at neighboring position for gradient)
    phi_other, _, _, _ = u_other
    
    # Select the appropriate velocity component based on spatial orientation
    if orientation == 1        # x-direction
        v_component = vx_mine
    elseif orientation == 2    # y-direction  
        v_component = vy_mine
    else  # orientation == 3   # z-direction
        v_component = vz_mine
    end
    
    # Return contributions to each equation
    # Note: negative sign from the PDE: -(ε vx ∂ϕ/∂x + ε vy ∂ϕ/∂y + ε vz ∂ϕ/∂z)
    # Assuming ε = 1 (porosity), modify if needed
    return SVector(v_component * phi_other,  # contribution to ϕ equation
                   zero(phi_other),           # no contribution to vx equation (auxiliary)
                   zero(phi_other),           # no contribution to vy equation (auxiliary)
                   zero(phi_other))           # no contribution to vz equation (auxiliary)
end


# Create a simple simulation setup to check everything works so fa


equation = NonconservativeLinearAdvectionEquation3D()

function initial_condition_test_open(x, t, equation::NonconservativeLinearAdvectionEquation3D, all_physical_parameters)
    #@show x
    (; ε) = all_physical_parameters
    phi = sin(x[2])
    advection_velocity_x = ε
    SVector(phi, 0.0, advection_velocity_x, 0.0)  # Assuming advection in x-direction only
end


initial_condition_test(x, t, equation::NonconservativeLinearAdvectionEquation3D) = initial_condition_test_open(x, t, equation::NonconservativeLinearAdvectionEquation3D, all_physical_parameters)


coordinates_min = (0.0, 0.0, 0.0)
coordinates_max = (2π, 2π, 2π)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000)


# Create a DGSEM solver with polynomials of degree `polydeg`
# Remember to pass a tuple of the form `(conservative_flux, nonconservative_flux)`
# as `surface_flux` and `volume_flux` when working with nonconservative terms
volume_flux = (flux_central, flux_nonconservative)
surface_flux = (flux_lax_friedrichs, flux_nonconservative)
solver = DGSEM(polydeg = 3, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

# Setup the spatial semidiscretization containing all ingredients
semi = SemidiscretizationHyperbolic(mesh, equation, initial_condition_test, solver)

# Create an ODE problem with given time span
tspan = (0.0, 2.1)
ode = semidiscretize(semi, tspan)



# Set up some standard callbacks summarizing the simulation setup and computing
# errors of the numerical solution
summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 50)
callbacks = CallbackSet(summary_callback, analysis_callback)

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes
# the passed callbacks
sol = solve(ode, Tsit5(), abstol = 1.0e-6, reltol = 1.0e-6;
            ode_default_options()..., callback = callbacks)


# Plot the numerical solution at the final time
pd = PlotData2D(sol)#,  slice=:y)
plot(pd["phi"])
plot!(getmesh(pd))

pd = PlotData1D(sol,  slice=:y)
plot(pd["phi"])
plot!(getmesh(pd))





# See the type hierarchy and structure
dump(Trixi.AbstractLinearScalarAdvectionEquation{2, 1})
dump(LinearScalarAdvectionEquation3D)
# Get methods defined for this type
methods(AbstractEquations)

# See all subtypes
subtypes(AbstractEquations)

# Get field names if it's a struct
fieldnames(AbstractEquations)

# Get more detailed type information
typeof(AbstractEquations)
supertype(LaplaceDiffusion2D)
