# load internal functions and structs we need
using Trixi: BoundaryConditionDoNothing, BoundaryConditionDirichlet, AbstractEquations

#  load internal functions and structs we need to modify / add functionality
import Trixi: varnames, default_analysis_integrals, flux, max_abs_speed_naive,
    have_nonconservative_terms


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
        v_component = vx_mine * Lx_inv_c
    elseif orientation == 2    # y-direction  
        v_component = vy_mine * Ly_inv_c
    else  # orientation == 3   # z-direction
        v_component = vz_mine * Lz_inv_c
    end

    # Return contributions to each equation
    return SVector(v_component * phi_other,  # contribution to ϕ equation
        zero(phi_other),           # no contribution to vx equation (auxiliary)
        zero(phi_other),           # no contribution to vy equation (auxiliary)
        zero(phi_other),           # no contribution to vz equation (auxiliary)
        zero(phi_other))           # no contribution to d equation (auxiliary)
end

@inline function (boundary_condition::BoundaryConditionDirichlet)(
    u_inner,
    orientation,
    direction,
    x, t,
    surface_flux_functions::Tuple,
    equations::DiffusionConvectionHyperbolic3D)

    # Call the boundary function to get all 5 values at this location and time
    u_boundary = boundary_condition.boundary_value_function(x, t, equations)

    surface_flux_function, nonconservative_flux_function = surface_flux_functions
    # Calculate boundary flux using the surface flux function
    if direction == 2  # u_inner is "left", u_boundary is "right"
        flux_conservative = surface_flux_function(u_inner, u_boundary, orientation, equations)
        flux_nonconservative = nonconservative_flux_function(u_inner, u_boundary, orientation, equations)
    else  # u_boundary is "left", u_inner is "right"  
        flux_conservative = surface_flux_function(u_boundary, u_inner, orientation, equations)
        flux_nonconservative = nonconservative_flux_function(u_boundary, u_inner, orientation, equations)
    end

    # In trixi you will often find the following quote. I THINK (not sure)
    # this is also needed here!

    # Note the factor 0.5 necessary for the nonconservative fluxes based on
    # the interpretation of global SBP operators coupled discontinuously via
    # central fluxes/SATs
    combined_flux = flux_conservative + 0.5f0 * flux_nonconservative
    # combined_flux = flux_conservative +  flux_nonconservative
    return combined_flux
end


@inline function (::BoundaryConditionDoNothing)(u_inner, orientation, direction::Integer, x, t,
                                                     surface_flux_functions::Tuple,
                                                     equations::DiffusionConvectionHyperbolic3D)
    
    surface_flux_function, nonconservative_flux_function = surface_flux_functions
    
    # For "do nothing" BC, we compute fluxes using only the interior state
    flux_conservative = surface_flux_function(u_inner, u_inner, orientation, equations)
    flux_nonconservative = nonconservative_flux_function(u_inner, u_inner, orientation, equations)
    
    # In trixi you will often find the following quote. I THINK (not sure)
    # this is also needed here!

    # Note the factor 0.5 necessary for the nonconservative fluxes based on
    # the interpretation of global SBP operators coupled discontinuously via
    # central fluxes/SATs
    combined_flux = flux_conservative + 0.5f0 * flux_nonconservative

    # combined_flux = flux_conservative +  flux_nonconservative
    
    return combined_flux
end





