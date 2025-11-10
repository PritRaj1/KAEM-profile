module WeightResamplers

export ResidualResampler, SystematicResampler, StratifiedResampler, resampler_map

using Random, Distributions, LinearAlgebra
using NNlib: softmax

using ..Utils

### Note: potential thread divergence on GPU for resampler searchsortedfirsts
function check_ESS(
        weights;
        ESS_threshold = 0.5f0,
    )
    """Effective sample size"""
    B, N = size(weights)
    ESS = dropdims(1 ./ sum(weights .^ 2, dims = 2); dims = 2)
    ESS_bool = ESS .< ESS_threshold * N
    resample_bool = any(ESS_bool)
    return ESS_bool, resample_bool, B, N
end

struct ResidualResampler <: AbstractResampler
    ESS_threshold::Float32
    _phantom::Bool
end

function residual_single(
        ESS_bool,
        cdf,
        u,
        integer_counts,
        N
    )
    deterministic_part = reduce(vcat, map(i -> fill(i, integer_counts[i]), 1:N))
    residual_part = dropdims(sum(1 .+ (cdf .< u'); dims = 1); dims = 1)
    residual_part = ifelse.(residual_part .> N, N, residual_part)
    return ifelse.(ESS_bool, vcat(deterministic_part, residual_part), 1:N)
end

function residual_kernel(
        ESS_bool,
        cdf,
        u,
        integer_counts,
        num_remaining,
        B,
        N,
    )
    return reduce(
        hcat,
        map(
            b -> residual_single(
                view(ESS_bool, b),
                view(cdf, b, (N - num_remaining[b] + 1):N),
                view(u, b, (N - num_remaining[b] + 1):N),
                view(integer_counts, b, :),
                N
            ), 1:B
        )
    )'
end

function (r::ResidualResampler)(
        weights;
        rng = Random.default_rng(),
    )
    """
    Residual resampling for weight filtering.

    Args:
        weights: The weights of the population.
        ESS_bool: A boolean array indicating if the ESS is above the threshold.
        rng: Random seed for reproducibility.

    Returns:
        - The resampled indices.
    """
    ESS_bool, resample_bool, B, N = check_ESS(
        weights;
        ESS_threshold = r.ESS_threshold,
    )

    # Number times to replicate each sample
    integer_counts = Int.(floor.(weights .* N))
    num_remaining = dropdims(N .- sum(integer_counts, dims = 2); dims = 2)

    # Residual weights to resample from
    residual_weights = softmax(weights .* (N .- integer_counts), dims = 2)

    # CDF and variate for resampling
    u = rand(rng, Float32, B, N)
    cdf = cumsum(residual_weights, dims = 2)
    return residual_kernel(ESS_bool, cdf, u, integer_counts, num_remaining, B, N)
end

struct SystematicResampler <: AbstractResampler
    ESS_threshold::Float32
    _phantom::Bool
end

function systematic_single(
        ESS_bool,
        cdf,
        u,
        N,
    )
    indices = dropdims(sum(1 .+ (cdf .< u'); dims = 1); dims = 1)
    indices = ifelse.(indices .> N, N, indices)
    return ifelse.(ESS_bool, indices, 1:N)
end

function systematic_kernel(
        ESS_bool,
        cdf,
        u,
        B,
        N,
    )
    return reduce(
        hcat,
        map(
            b -> systematic_single(
                view(ESS_bool, b),
                view(cdf, b, :),
                view(u, b, :),
                N
            ), 1:B
        )
    )'
end

function (r::SystematicResampler)(
        weights;
        rng = Random.default_rng(),
    )
    """
    Systematic resampling for weight filtering.

    Args:
        weights: The weights of the population.
        ESS_bool: A boolean array indicating if the ESS is above the threshold.
        rng: Random seed for reproducibility.

    Returns:
        - The resampled indices.
    """
    ESS_bool, resample_bool, B, N = check_ESS(weights; ESS_threshold = r.ESS_threshold)

    cdf = cumsum(weights, dims = 2)

    # Systematic thresholds
    u = (rand(rng, Float32, B, 1) .+ (0:(N - 1))') ./ N
    return systematic_kernel(ESS_bool, cdf, u, B, N)
end

struct StratifiedResampler <: AbstractResampler
    ESS_threshold::Float32
    _phantom::Bool
end

function (r::StratifiedResampler)(
        weights;
        rng = Random.default_rng(),
    )
    """
    Systematic resampling for weight filtering.

    Args:
        weights: The weights of the population.
        ESS_bool: A boolean array indicating if the ESS is above the threshold.
        rng: Random seed for reproducibility.

    Returns:
        - The resampled indices.
    """
    ESS_bool, resample_bool, B, N = check_ESS(weights; ESS_threshold = r.ESS_threshold)

    cdf = cumsum(weights, dims = 2)

    # Stratified thresholds
    u = (rand(rng, Float32, B, N) .+ (0:(N - 1))') ./ N
    return systematic_kernel(ESS_bool, cdf, u, B, N)
end

const resampler_map::Dict{String, Function} = Dict(
    "residual" => (ESS_threshold, verbose) -> ResidualResampler(ESS_threshold, verbose),
    "systematic" => (ESS_threshold, verbose) -> SystematicResampler(ESS_threshold, verbose),
    "stratified" => (ESS_threshold, verbose) -> StratifiedResampler(ESS_threshold, verbose)
)
end
