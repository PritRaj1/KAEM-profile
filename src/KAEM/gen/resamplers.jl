module WeightResamplers

export ResidualResampler, SystematicResampler, StratifiedResampler, resampler_map

using Random, Distributions, LinearAlgebra
using NNlib: softmax

using ..Utils

### Note: potential thread divergence on GPU for resampler searchsortedfirsts
function check_ESS(
        weights;
        ESS_threshold = 0.5f0,
        verbose = false,
    )
    """
    Filter the latent variable for a index of the Steppingstone sum using residual resampling.

    Args:
        logllhood: A matrix of log-likelihood values.
        weights: The weights of the population.
        t_resample: The temperature at which the last resample occurred.
        t2: The temperature at which to update the weights.
        rng: Random seed for reproducibility.
        ESS_threshold: The threshold for the effective sample size.
        resampler: The resampling function.

    Returns:
        - The resampled indices.    
    """
    B, N = size(weights)

    # Check effective sample size
    ESS = dropdims(1 ./ sum(weights .^ 2, dims = 2); dims = 2)
    ESS_bool = ESS .< ESS_threshold * N
    resample_bool = any(ESS_bool)

    # Only resample when needed
    verbose && (resample_bool && println("Resampling!"))
    return ESS_bool, resample_bool, B, N
end

struct ResidualResampler <: AbstractResampler
    ESS_threshold::Float32
    verbose::Bool
end

function residual_single(
        ESS_bool,
        cdf,
        u,
        integer_counts,
        N,
    )
    !ESS_bool && return collect(1:N)
    deterministic_part = reduce(vcat, map(i -> fill(i, integer_counts[i]), 1:N))
    residual_part = reduce(vcat, searchsortedfirst.(Ref(cdf), u))
    return vcat(deterministic_part, clamp.(residual_part, 1, N))
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
                ESS_bool[b],
                cdf[b, (N - num_remaining[b] + 1):N],
                u[b, (N - num_remaining[b] + 1):N],
                integer_counts[b, :], N
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
        verbose = r.verbose,
    )
    !resample_bool && return repeat(collect(1:N)', B, 1)

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
    verbose::Bool
end

function systematic_single(
        ESS_bool,
        cdf,
        u,
        N,
    )
    !ESS_bool && return collect(1:N)
    indices = searchsortedfirst.(Ref(cdf), u)
    return clamp.(indices, 1, N)
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
        map(b -> systematic_single(ESS_bool[b], cdf[b, :], u[b, :], N), 1:B)
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
    ESS_bool, resample_bool, B, N = check_ESS(weights; ESS_threshold = r.ESS_threshold, verbose = r.verbose)
    !resample_bool && return repeat(collect(1:N)', B, 1)

    cdf = cumsum(weights, dims = 2)

    # Systematic thresholds
    u = (rand(rng, Float32, B, 1) .+ (0:(N - 1))') ./ N
    return systematic_kernel(ESS_bool, cdf, u, B, N)
end

struct StratifiedResampler <: AbstractResampler
    ESS_threshold::Float32
    verbose::Bool
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
    ESS_bool, resample_bool, B, N = check_ESS(weights; ESS_threshold = r.ESS_threshold, verbose = r.verbose)
    !resample_bool && return repeat(collect(1:N)', B, 1)

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
