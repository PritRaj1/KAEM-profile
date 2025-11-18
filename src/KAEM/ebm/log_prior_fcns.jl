module LogPriorFCNs

export LogPriorULA, LogPriorMix, LogPriorUnivariate

using NNlib: logsoftmax, softmax
using LinearAlgebra, Accessors, Random, ComponentArrays

using ..Utils
using ..EBM_Model

function log_norm(norm, ε)
    norm = sum(norm; dims = 3)
    log_norm = @. log(norm + ε)
    return dropdims(log_norm, dims = 3)
end

function log_mix_pdf(
        f,
        α,
        π_0,
        Z,
        ε,
        Q,
        P,
        S,
    )
    exp_f = @. exp(f) * π_0 * α / Z
    lp = dropdims(sum(exp_f; dims = 2); dims = 2)
    return log.(dropdims(prod(lp; dims = 1) .+ ε; dims = 1))
end

struct LogPriorULA{T <: Float32} <: AbstractLogPrior
    ε::T
end

struct LogPriorUnivariate{T <: Float32} <: AbstractLogPrior
    ε::T
    normalize::Bool
end

struct LogPriorMix{T <: Float32} <: AbstractLogPrior
    ε::T
    normalize::Bool
end

function (lp::LogPriorULA)(
        z,
        ebm,
        ps,
        st_kan,
        st_lyrnorm,
    )
    log_π0 = dropdims(
        sum(ebm.π_pdf(z, ps.dist.π_μ, ps.dist.π_σ; log_bool = true); dims = 1),
        dims = (1, 2),
    )
    f, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, dropdims(z; dims = 2))
    return dropdims(sum(f; dims = 1); dims = 1) + log_π0, st_lyrnorm_new
end

function (lp::LogPriorUnivariate)(
        z,
        ebm,
        ps,
        st_kan,
        st_lyrnorm;
        ula = false,
    )
    """
    The log-probability of the ebm-prior.

    ∑_q [ ∑_p f_{q,p}(z_qp) ]

    Args:
        ebm: The ebm-prior.
        z: The component-wise latent samples to evaulate the measure on, (num_samples, q)
        ps: The parameters of the ebm-prior.
        st: The states of the ebm-prior.
        normalize: Whether to normalize the log-probability.
        ε: The small value to avoid log(0).
        agg: Whether to sum the log-probability over the samples.

    Returns:
        The unnormalized log-probability of the ebm-prior.
        The updated states of the ebm-prior.
    """
    Q, P = ebm.q_size, ebm.p_size
    log_π0 = ebm.π_pdf(z, ps.dist.π_μ, ps.dist.π_σ; log_bool = true)

    log_π0 =
        lp.normalize && !ula ?
        log_π0 .- log_norm(first(ebm.quad(ebm, ps, st_kan, st_lyrnorm)), lp.ε) : log_π0

    f, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, reshape(z, P, :))
    f_4d = reshape(f, Q, Q, P, :)
    I_q = Array{Float32}(I, Q, Q) |> pu

    f_diag = dropdims(sum(f_4d .* I_q; dims = 2); dims = 2)
    log_p = dropdims(sum(f_diag .+ log_π0; dims = (1, 2)); dims = (1, 2))
    return log_p, st_lyrnorm_new
end

function dotprod_attn(
        Q,
        K,
        z,
    )
    scale = sqrt(Float32(size(z)[end]))
    return dropdims(sum(Q .* z .* K .* z; dims = 3); dims = 3) ./ scale
end

function (lp::LogPriorMix)(
        z,
        ebm,
        ps,
        st_kan,
        st_lyrnorm;
        ula = false,
    )
    """
    The log-probability of the mixture ebm-prior.

    ∑_q [ log ( ∑_p α_p exp(f_{q,p}(z_q)) π_0(z_q) ) ]


    Args:
        mix: The mixture ebm-prior.
        z: The component-wise latent samples to evaulate the measure on, (num_samples, q)
        ps: The parameters of the mixture ebm-prior.
        st: The states of the mixture ebm-prior.
        normalize: Whether to normalize the log-probability.
        ε: The small value to avoid log(0).

    Returns:
        The unnormalized log-probability of the mixture ebm-prior.
        The updated states of the mixture ebm-prior.
    """
    Q, P = ebm.q_size, ebm.p_size
    alpha =
        ebm.bool_config.use_attention_kernel ?
        dotprod_attn(ps.attention.Q, ps.attention.K, z) :
        (ebm.bool_config.train_props ? ps.dist.α : pu(ones(Float32, Q, P)))

    alpha = softmax(alpha; dims = 2)
    π_0 = ebm.π_pdf(z, ps.dist.π_μ, ps.dist.π_σ; log_bool = false)

    # Energy functions of each component, q -> p
    f, st_lyrnorm = ebm(ps, st_kan, st_lyrnorm, dropdims(z; dims = 2))
    Z =
        lp.normalize && !ula ?
        dropdims(sum(first(ebm.quad(ebm, ps, st_kan, st_lyrnorm)), dims = 3), dims = 3) :
        ones(Float32, Q, P) |> pu

    reg = ebm.λ > 0 ? ebm.λ * sum(abs.(alpha)) : 0.0f0
    log_p = log_mix_pdf(f, alpha, π_0, Z, lp.ε, Q, P, :)
    return log_p .+ reg, st_lyrnorm
end

end
