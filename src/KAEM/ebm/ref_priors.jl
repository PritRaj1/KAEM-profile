module RefPriors

export prior_map,
    UniformPrior, GaussianPrior, LogNormalPrior, LearnableGaussianPrior, EbmPrior

using CUDA, Lux, KernelAbstractions, Tullio

using ..Utils

struct UniformPrior{T <: Float32} <: AbstractPrior
    ε::T
end
struct GaussianPrior{T <: Float32} <: AbstractPrior
    ε::T
end
struct LogNormalPrior{T <: Float32} <: AbstractPrior
    ε::T
end
struct LearnableGaussianPrior{T <: Float32} <: AbstractPrior
    ε::T
end
struct EbmPrior{T <: Float32} <: AbstractPrior
    ε::T
end

function stable_log(pdf::AbstractArray{T, 3}, ε::T)::AbstractArray{T, 3} where {T <: Float32}
    return log.(pdf .+ ε)
end

function (prior::UniformPrior)(
        z::AbstractArray{T, 3},
        π_μ::AbstractArray{T, 1},
        π_σ::AbstractArray{T, 1};
        log_bool::Bool = false,
    )::AbstractArray{T, 3} where {T <: Float32}
    @tullio pdf[q, p, s] := (z[q, p, s] >= 0) * (z[q, p, s] <= 1)
    log_bool && return stable_log(pdf, prior.ε)
    return pdf
end

function (prior::GaussianPrior)(
        z::AbstractArray{T, 3},
        π_μ::AbstractArray{T, 1},
        π_σ::AbstractArray{T, 1};
        log_bool::Bool = false,
    )::AbstractArray{T, 3} where {T <: Float32}
    scale = T(1 / sqrt(2π))
    @tullio pdf[q, p, s] := exp(-z[q, p, s]^2 / 2)
    pdf = scale .* pdf
    log_bool && return stable_log(pdf, prior.ε)
    return pdf
end

function (prior::LogNormalPrior)(
        z::AbstractArray{T, 3},
        π_μ::AbstractArray{T, 1},
        π_σ::AbstractArray{T, 1};
        log_bool::Bool = false,
    )::AbstractArray{T, 3} where {T <: Float32}
    sqrt_2π = T(sqrt(2π))
    denom = z .* sqrt_2π .+ prior.ε
    z_eps = z .+ prior.ε
    @tullio pdf[q, p, s] := exp(-((log(z_eps[q, p, s]))^2) / 2) / denom[q, p, s]
    log_bool && return stable_log(pdf, prior.ε)
    return pdf
end

function (prior::LearnableGaussianPrior)(
        z::AbstractArray{T, 3},
        π_μ::AbstractArray{T, 1},
        π_σ::AbstractArray{T, 1};
        log_bool::Bool = false,
    )::AbstractArray{T, 3} where {T <: Float32}
    π_eps = π_σ .* T(sqrt(2π)) .+ prior.ε
    denom_eps = 2 .* π_σ .^ 2 .+ prior.ε
    @tullio pdf[q, p, s] :=
        1 / (abs(π_eps[p]) * exp(-((z[q, p, s] - π_μ[p])^2) / denom_eps[p]))
    log_bool && return stable_log(pdf, prior.ε)
    return pdf
end

function (prior::EbmPrior)(
        z::AbstractArray{T, 3},
        π_μ::AbstractArray{T, 1},
        π_σ::AbstractArray{T, 1};
        log_bool::Bool = false,
    )::AbstractArray{T, 3} where {T <: Float32}
    log_pdf = 0.0f0 .* z
    log_bool && return log_pdf
    return log_pdf .+ 1.0f0
end

const prior_map = Dict(
    "uniform" => ε -> UniformPrior(ε),
    "gaussian" => ε -> GaussianPrior(ε),
    "lognormal" => ε -> LogNormalPrior(ε),
    "learnable_gaussian" => ε -> LearnableGaussianPrior(ε),
    "ebm" => ε -> EbmPrior(ε),
)

end
