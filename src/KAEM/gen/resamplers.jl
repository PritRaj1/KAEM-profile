module WeightResamplers

export ResidualResampler, SystematicResampler, StratifiedResampler, resampler_map

using Random, Distributions, LinearAlgebra
using NNlib: softmax
using Reactant: @allowscalar

using ..Utils

### Note: potential thread divergence on GPU for resampler searchsortedfirsts
function check_ESS(
        weights;
        ESS_threshold = 0.5f0,
    )
    """Effective sample size"""
    B, N = size(weights)
    ESS = 1 ./ sum(weights .^ 2, dims = 2)
    ESS_bool = ESS .< ESS_threshold * N
    return ESS_bool, B, N
end

struct ResidualResampler <: AbstractResampler
    ESS_threshold::Float32
    _phantom::Bool
end

function deterministic_single(
        integer_counts,
        N
    )
    """Replicates integer_counts times using stableho-compatible logic"""
    c = cumsum(integer_counts)
    L = sum(c)
    deterministic_part = 1 .+ dropdims(sum(c .< (1:N)'; dims = 1); dims = 1)
    return deterministic_part
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
    early_return = (1 .- ESS_bool) .* (1:N)'
    mask = (1:N)' .>= (N .- num_remaining .+ 1) # Whether allcoated residual or not

    residual_part = dropdims(sum(1 .+ (cdf .< reshape(u, B, 1, N)); dims = 2); dims = 2)
    residual_part = ifelse.(residual_part .> N, N, residual_part)

    deterministic_part = reduce(
        hcat,
        map(
            b -> deterministic_single(
                view(integer_counts, b, :),
                N
            ), 1:B
        )
    )'

    indices = (mask .* residual_part) .+ (1 .- mask) .* deterministic_part
    return early_return .+ ESS_bool .* indices
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
    ESS_bool, B, N = check_ESS(
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

function systematic_kernel(
        ESS_bool,
        cdf,
        u,
        B,
        N,
    )
    early_return = (1 .- ESS_bool) .* (1:N)'
    indices = dropdims(sum(1 .+ (cdf .< reshape(u, B, 1, N)); dims = 2); dims = 2)
    indices = ifelse.(indices .> N, N, indices)
    return early_return .+ ESS_bool .* indices
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
    ESS_bool, B, N = check_ESS(weights; ESS_threshold = r.ESS_threshold)

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
    ESS_bool, B, N = check_ESS(weights; ESS_threshold = r.ESS_threshold)

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
