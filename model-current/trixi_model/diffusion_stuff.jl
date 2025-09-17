# load internal functions and structs we need
using Trixi: AbstractEquationsParabolic, BoundaryConditionDirichlet, BoundaryConditionConstantNeumann, Gradient, Divergence 

#  load internal functions and structs we need to modify / add functionality
import Trixi: varnames, flux

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

@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner,
    u_inner,
    normal::AbstractVector,
    x, t,
    operator_type::Gradient,
    equations_parabolic::DiffusionConvectionParabolic3D)
    return boundary_condition.boundary_value_function(x, t, equations_parabolic)
end



@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner,
    u_inner,
    normal::AbstractVector,
    x, t,
    operator_type::Divergence,
    equations_parabolic::DiffusionConvectionParabolic3D)
    return flux_inner
end


@inline function (boundary_condition::BoundaryConditionNeumann)(flux_inner, u_inner,
                                                                normal::AbstractVector,
                                                                x, t,
                                                                operator_type::Divergence,
                                                                equations_parabolic::DiffusionConvectionParabolic3D)
    return boundary_condition.boundary_normal_flux_function(x, t, equations_parabolic)
end

@inline function (boundary_condition::BoundaryConditionNeumann)(flux_inner, u_inner,
                                                                normal::AbstractVector,
                                                                x, t,
                                                                operator_type::Gradient,
                                                                equations_parabolic::DiffusionConvectionParabolic3D)
    return flux_inner
end
