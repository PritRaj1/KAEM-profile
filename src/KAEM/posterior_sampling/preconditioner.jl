module Preconditioning

export init_mass_matrix, sample_momentum

using LinearAlgebra, Random, Distributions, Statistics

using ..Utils

abstract type Preconditioner end

struct IdentityPreconditioner <: Preconditioner end
struct DiagonalPreconditioner <: Preconditioner end

struct MixDiagonalPreconditioner{TR <: Real} <: Preconditioner
    p0::TR  # Proportion of zeros
    p1::TR  # Proportion of ones

    function MixDiagonalPreconditioner(p0::TR, p1::TR) where {TR <: Real}
        zero(TR) ≤ p0 + p1 ≤ one(TR) || throw(ArgumentError("p0+p1 < 0 or p0+p1 > 1"))
        return new{TR}(p0, p1)
    end
end

MixDiagonalPreconditioner() = MixDiagonalPreconditioner(1 // 3, 1 // 3)

# Default behavior - no preconditioning
function build_preconditioner!(
        dest::AbstractArray{T, 2},
        ::IdentityPreconditioner,
        std_devs::AbstractArray{T, 2};
        rng::AbstractRNG = Random.default_rng(),
    )::Nothing where {T <: Float32}
    fill!(dest, one(T))
    return nothing
end

# Diagonal preconditioning
function build_preconditioner!(
        dest::AbstractArray{T, 2},
        ::DiagonalPreconditioner,
        std_devs::AbstractArray{T, 2};
        rng::AbstractRNG = Random.default_rng(),
    )::Nothing where {T <: Float32}
    @. dest = ifelse(iszero(std_devs), one(T), one(T) / std_devs)
    return nothing
end

# Mixed diagonal preconditioning
function build_preconditioner!(
        dest::AbstractArray{T, 2},
        prec::MixDiagonalPreconditioner,
        std_devs::AbstractArray{T, 2};
        rng::AbstractRNG = Random.default_rng(),
    )::Nothing where {T <: Float32}
    u = rand(rng, T)

    if u ≤ prec.p0
        # Use inverse standard deviations
        @. dest = ifelse(iszero(std_devs), one(T), one(T) / std_devs)
    elseif u ≤ prec.p0 + prec.p1
        # Use identity
        fill!(dest, one(T))
    else
        # Random mixture
        mix = rand(rng, T)
        rmix = one(T) - mix
        @. dest = ifelse(iszero(std_devs), one(T), mix + rmix / std_devs)
    end
    return nothing
end

function init_mass_matrix(
        z::AbstractArray{T, 3},
        rng::AbstractRNG = Random.default_rng(),
    )::AbstractArray{T, 2} where {T <: Float32}
    Σ = sum((z .- mean(z; dims = 3)) .^ 2; dims = 3) ./ (size(z, 3) - 1) # Diagonal Covariance
    β = rand(rng, Truncated(Beta(1, 1), 0.5, 2 / 3)) |> T
    @. Σ = sqrt(β * (1 / Σ) + (1 - β)) # Augmented mass matrix
    return dropdims(Σ; dims = 3)
end

# This is transformed momentum!
function sample_momentum(
        z::AbstractArray{T, 3},
        M::AbstractArray{T, 2};
        rng::AbstractRNG = Random.default_rng(),
        preconditioner::Preconditioner = MixDiagonalPreconditioner(),
    )::Tuple{AbstractArray{T, 3}, AbstractArray{T, 2}} where {T <: Float32}

    # Initialize M^{1/2}
    Σ = sqrt.(sum((z .- mean(z; dims = 3)) .^ 2; dims = 3) ./ (size(z, 3) - 1))
    build_preconditioner!(M, preconditioner, dropdims(Σ; dims = 3); rng = rng)

    # Sample y ~ N(0,I) directly (transformed momentum)
    y = randn(rng, T, size(z)) |> pu
    return y, M
end

end
