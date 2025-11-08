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

function stable_log(pdf, ε)
    return log.(pdf .+ ε)
end

function (prior::UniformPrior)(
        z,
        π_μ,
        π_σ;
        log_bool = false,
    )
    @tullio pdf[q, p, s] := (z[q, p, s] >= 0) * (z[q, p, s] <= 1)
    log_bool && return stable_log(pdf, prior.ε)
    return pdf
end

function (prior::GaussianPrior)(
        z,
        π_μ,
        π_σ;
        log_bool = false,
    )
    scale = Float32(1 / sqrt(2π))
    @tullio pdf[q, p, s] := exp(-z[q, p, s]^2 / 2)
    pdf = scale .* pdf
    log_bool && return stable_log(pdf, prior.ε)
    return pdf
end

function (prior::LogNormalPrior)(
        z,
        π_μ,
        π_σ;
        log_bool = false,
    )
    sqrt_2π = Float32(sqrt(2π))
    denom = z .* sqrt_2π .+ prior.ε
    z_eps = z .+ prior.ε
    @tullio pdf[q, p, s] := exp(-((log(z_eps[q, p, s]))^2) / 2) / denom[q, p, s]
    log_bool && return stable_log(pdf, prior.ε)
    return pdf
end

function (prior::LearnableGaussianPrior)(
        z,
        π_μ,
        π_σ;
        log_bool = false,
    )
    π_eps = π_σ .* Float32(sqrt(2π)) .+ prior.ε
    denom_eps = 2 .* π_σ .^ 2 .+ prior.ε
    @tullio pdf[q, p, s] :=
        1 / (abs(π_eps[p]) * exp(-((z[q, p, s] - π_μ[p])^2) / denom_eps[p]))
    log_bool && return stable_log(pdf, prior.ε)
    return pdf
end

function (prior::EbmPrior)(
        z,
        π_μ,
        π_σ;
        log_bool = false,
    )
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
