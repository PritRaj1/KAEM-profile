module LangevinUpdates

export update_z!, logpos_withgrad, leapfrog

using Lux, ComponentArrays, Accessors

using ..Utils
using ..KAEM_model

include("log_posteriors.jl")
using .LogPosteriors: autoMALA_value_and_grad

## ULA ##
function update_z!(
        z,
        ∇z,
        η::T,
        ξ,
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
        z,
        momentum, # p*
        ∇z,
        M,
        η,
    )
    η = reshape(η, 1, 1, length(η))
    @. momentum = momentum + (η / 2) * ∇z / M
    @. z = z + η * momentum / M
    return nothing
end

function momentum_update!(
        momentum, # p*
        ∇ẑ,
        M,
        η,
    )
    η = reshape(η, 1, 1, length(η))
    @. momentum = momentum + (η / 2) * ∇ẑ / M
    return nothing
end

function logpos_withgrad(
        z,
        ∇z,
        x,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
    )
    logpos, ∇z_k, st_ebm, st_gen =
        autoMALA_value_and_grad(z, ∇z, x, temps, model, ps, st_kan, st_lux)
    @reset st_lux.ebm = st_ebm
    @reset st_lux.gen = st_gen

    return logpos,
        ∇z_k,
        st_lux
end

function leapfrog(
        z,
        ∇z,
        x,
        temps,
        logpos_z,
        p, # This is momentum = M^{-1/2}p
        M, # This is M^{1/2}
        η,
        model,
        ps,
        st_kan,
        st_lux,
    )
    """
    Implements preconditioned Hamiltonian dynamics with transformed momentum:
    y*(x,y)   = y  + (eps/2)M^{-1/2}grad(log pi)(x)
    x'(x,y*)  = x  + eps M^{-1/2}y*
    y'(x',y*) = y* + (eps/2)M^{-1/2}grad(log pi)(x')
    """
    Q, P, S = sizez

    # Half-step momentum update (p* = p + (eps/2)M^{-1/2}grad) and full step position update
    momentum = copy(p)
    position_update!(z, p, ∇z, M, η)

    # Get gradient at new position
    logpos_ẑ, ∇ẑ, st_lux =
        logpos_withgrad(z, ∇z, x, temps, model, ps, st_kan, st_lux)

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
