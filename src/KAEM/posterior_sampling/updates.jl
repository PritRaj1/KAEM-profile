module LangevinUpdates

export update_z!, logpos_withgrad, leapfrog

using Lux, ComponentArrays, Accessors

using ..Utils
using ..T_KAM_model

include("log_posteriors.jl")
using .LogPosteriors: autoMALA_value_and_grad

## ULA ##
function update_z!(
        z::AbstractArray{T, 3},
        ∇z::AbstractArray{T, 3},
        η::T,
        ξ::AbstractArray{T, 3},
        sqrt_2η::T,
        Q::Int,
        P::Int,
        S::Int,
    )::Nothing where {T <: Float32}
    @. z = z + η * ∇z + sqrt_2η * ξ
    return nothing
end

## autoMALA ##
function position_update!(
        z::AbstractArray{T, 3},
        momentum::AbstractArray{T, 3}, # p*
        ∇z::AbstractArray{T, 3},
        M::AbstractArray{T, 2},
        η::AbstractArray{T, 1},
    )::Nothing where {T <: Float32}
    η = reshape(η, 1, 1, length(η))
    @. momentum = momentum + (η / 2) * ∇z / M
    @. z = z + η * momentum / M
    return nothing
end

function momentum_update!(
        momentum::AbstractArray{T, 3}, # p*
        ∇ẑ::AbstractArray{T, 3},
        M::AbstractArray{T, 2},
        η::AbstractArray{T, 1},
    )::Nothing where {T <: Float32}
    η = reshape(η, 1, 1, length(η))
    @. momentum = momentum + (η / 2) * ∇ẑ / M
    return nothing
end

function logpos_withgrad(
        z::AbstractArray{T, 3},
        ∇z::AbstractArray{T, 3},
        x::AbstractArray{T},
        temps::AbstractArray{T, 1},
        model::T_KAM{T},
        ps::ComponentArray{T},
        st_kan::ComponentArray{T},
        st_lux::NamedTuple,
    )::Tuple{
        AbstractArray{T, 1},
        AbstractArray{T, 3},
        NamedTuple,
    } where {T <: Float32}
    logpos, ∇z_k, st_ebm, st_gen =
        autoMALA_value_and_grad(z, ∇z, x, temps, model, ps, st_kan, st_lux)
    @reset st_lux.ebm = st_ebm
    @reset st_lux.gen = st_gen

    return logpos,
        ∇z_k,
        st_lux
end

function leapfrog(
        z::AbstractArray{T, 3},
        ∇z::AbstractArray{T, 3},
        x::AbstractArray{T},
        temps::AbstractArray{T, 1},
        logpos_z::AbstractArray{T, 1},
        p::AbstractArray{T, 3}, # This is momentum = M^{-1/2}p
        M::AbstractArray{T, 2}, # This is M^{1/2}
        η::AbstractArray{T, 1},
        model::T_KAM{T},
        ps::ComponentArray{T},
        st_kan::ComponentArray{T},
        st_lux::NamedTuple,
    )::Tuple{
        AbstractArray{T, 3},
        AbstractArray{T, 1},
        AbstractArray{T, 3},
        AbstractArray{T, 3},
        AbstractArray{T, 1},
        NamedTuple,
    } where {T <: Float32}
    """
    Implements preconditioned Hamiltonian dynamics with transformed momentum:
    y*(x,y)   = y  + (eps/2)M^{-1/2}grad(log pi)(x)
    x'(x,y*)  = x  + eps M^{-1/2}y*
    y'(x',y*) = y* + (eps/2)M^{-1/2}grad(log pi)(x')
    """
    Q, P, S = size(z)

    # Half-step momentum update (p* = p + (eps/2)M^{-1/2}grad) and full step position update
    momentum = copy(p)
    position_update!(z, p, ∇z, M, η)

    # Get gradient at new position
    logpos_ẑ, ∇ẑ, st_lux =
        logpos_withgrad((z), (∇z), x, temps, model, ps, st_kan, st_lux)

    # Half-step momentum update (p* = p + (eps/2)M^{-1/2}grad)
    momentum_update!(p, ∇ẑ, M, η)

    # Hamiltonian difference for transformed momentum
    # H(x,y) = -log(pi(x)) + (1/2)||p||^2 since p ~ N(0,I)
    log_r =
        logpos_ẑ - logpos_z -
        dropdims(
        sum(p .^ 2; dims = (1, 2)) - sum(momentum .^ 2; dims = (1, 2));
        dims = (1, 2),
    ) ./ 2

    return z, logpos_ẑ, ∇ẑ, -p, log_r, st_lux
end

end
