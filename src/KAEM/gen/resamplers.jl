module WeightResamplers

export ResidualResampler, SystematicResampler, StratifiedResampler, resampler_map

using CUDA, Random, Distributions, LinearAlgebra, ParallelStencil
using NNlib: softmax

using ..Utils

### Note: potential thread divergence on GPU for resampler searchsortedfirsts
@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, Float32, 3)
else
    @init_parallel_stencil(Threads, Float32, 3)
end

function check_ESS(
        weights::AbstractArray{T, 2};
        ESS_threshold::T = 0.5f0,
        verbose::Bool = false,
    )::Tuple{AbstractArray{Bool, 1}, Bool, Int, Int} where {T <: Float32}
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

@parallel_indices (b) function residual_kernel!(
        idxs::AbstractArray{T, 2},
        ESS_bool::AbstractArray{Bool, 1},
        cdf::AbstractArray{T, 2},
        u::AbstractArray{T, 2},
        num_remaining::AbstractArray{Int, 1},
        integer_counts::AbstractArray{Int, 2},
        B::Int,
        N::Int,
    )::Nothing where {T <: Float32}
    c = 1

    if !ESS_bool[b] # No resampling
        for n in 1:N
            idxs[b, n] = n
        end
    else

        # Deterministic replication
        for s in 1:N
            count = integer_counts[b, s]
            if count > 0
                for i in c:(c + count - 1)
                    idxs[b, i] = s
                end
                c += count
            end
        end

        # Multinomial resampling
        if num_remaining[b] > 0
            for k in 1:num_remaining[b]
                idx = N
                for j in 1:N
                    if cdf[b, j] >= u[b, k]
                        idx = j
                        break
                    end
                end
                idx = idx > N ? N : idx
                idxs[b, c] = idx
                c += 1
            end
        end
    end
    return nothing
end

function (r::ResidualResampler)(
        weights::AbstractArray{T, 2};
        rng::AbstractRNG = Random.default_rng(),
    )::AbstractArray{Int, 2} where {T <: Float32}
    """
    Residual resampling for weight filtering.

    Args:
        weights: The weights of the population.
        ESS_bool: A boolean array indicating if the ESS is above the threshold.
        rng: Random seed for reproducibility.

    Returns:
        - The resampled indices.
    """
    ESS_bool, resample_bool, B, N = check_ESS(weights; ESS_threshold = r.ESS_threshold, verbose = r.verbose)
    !resample_bool && return repeat(collect(1:N)', B, 1)

    # Number times to replicate each sample
    integer_counts = Int.(floor.(weights .* N))
    num_remaining = dropdims(N .- sum(integer_counts, dims = 2); dims = 2)

    # Residual weights to resample from
    residual_weights = softmax(weights .* (N .- integer_counts), dims = 2)

    # CDF and variate for resampling
    u = pu(rand(rng, T, B, N))
    cdf = cumsum(residual_weights, dims = 2)

    idxs = @zeros(B, N)
    @parallel (1:B) residual_kernel!(
        idxs,
        ESS_bool,
        cdf,
        u,
        num_remaining,
        integer_counts,
        B,
        N,
    )
    return Int.(idxs)
end

struct SystematicResampler <: AbstractResampler
    ESS_threshold::Float32
    verbose::Bool
end

@parallel_indices (b) function systematic_kernel!(
        idxs::AbstractArray{T, 2},
        ESS_bool::AbstractArray{Bool, 1},
        cdf::AbstractArray{T, 2},
        u::AbstractArray{T, 2},
        B::Int,
        N::Int,
    )::Nothing where {T <: Float32}
    if !ESS_bool[b] # No resampling
        for n in 1:N
            idxs[b, n] = n
        end
    else
        # Searchsortedfirst
        for n in 1:N
            idx = N
            for j in 1:N
                if cdf[b, j] >= u[b, n]
                    idx = j
                    break
                end
            end
            idx = idx > N ? N : idx
            idxs[b, n] = idx
        end
    end
    return nothing
end

function (r::SystematicResampler)(
        weights::AbstractArray{T, 2};
        rng::AbstractRNG = Random.default_rng(),
    )::AbstractArray{Int, 2} where {T <: Float32}
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
    u = pu((rand(rng, T, B, 1) .+ (0:(N - 1))') ./ N)

    idxs = @zeros(B, N)
    @parallel (1:B) systematic_kernel!(idxs, ESS_bool, cdf, u, B, N)
    return Int.(idxs)
end

struct StratifiedResampler <: AbstractResampler
    ESS_threshold::Float32
    verbose::Bool
end

function (r::StratifiedResampler)(
        weights::AbstractArray{T, 2};
        rng::AbstractRNG = Random.default_rng(),
    )::AbstractArray{Int, 2} where {T <: Float32}
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
    u = pu((rand(rng, T, B, N) .+ (0:(N - 1))') ./ N)

    idxs = @zeros(B, N)
    @parallel (1:B) systematic_kernel!(idxs, ESS_bool, cdf, u, B, N)
    return Int.(idxs)
end

const resampler_map::Dict{String, Function} = Dict(
    "residual" => (ESS_threshold, verbose) -> ResidualResampler(ESS_threshold, verbose),
    "systematic" => (ESS_threshold, verbose) -> SystematicResampler(ESS_threshold, verbose),
    "stratified" => (ESS_threshold, verbose) -> StratifiedResampler(ESS_threshold, verbose)
)
end
