module ImportanceSampling

using CUDA, ComponentArrays, Random, Enzyme, Reactant
using Statistics, Lux, LuxCUDA
using NNlib: softmax

export ImportanceLoss, initialize_importance_loss

using ..Utils
using ..T_KAM_model

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_IS

function accumulator(
        weights,
        logprior,
        logllhood,
    )
    return weights' * (logprior + logllhood)
end

function loss_accum(
        weights_resampled,
        logprior,
        logllhood,
        resampled_idxs,
        B,
        S,
    )

    loss = 0.0f0
    for b in 1:B
        loss =
            loss + accumulator(
            weights_resampled[b, :],
            logprior[resampled_idxs[b, :]],
            logllhood[b, resampled_idxs[b, :]],
        )
    end

    return loss / B
end

function sample_importance(
        ps::ComponentArray{T},
        st_kan::ComponentArray{T},
        st_lux::NamedTuple,
        m::T_KAM{T},
        x::AbstractArray{T};
        rng::AbstractRNG = Random.default_rng(),
    )::Tuple{
        AbstractArray{T, 3},
        AbstractArray{T, 3},
        NamedTuple,
        NamedTuple,
        AbstractArray{T, 2},
        AbstractArray{Int, 2},
        AbstractArray{T},
    } where {T <: Float32}

    # Prior is proposal for importance sampling
    z_posterior, st_lux_ebm = m.sample_prior(m, m.IS_samples, ps, st_kan, st_lux, rng)
    noise = pu(randn(rng, T, m.lkhood.x_shape..., size(z_posterior)[end], size(x)[end]))
    logllhood, st_lux_gen = log_likelihood_IS(
        z_posterior,
        x,
        m.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux.gen,
        noise;
        ε = m.ε,
    )

    # Posterior weights and resampling
    weights = softmax(logllhood, dims = 2)
    resampled_idxs = m.lkhood.resample_z(weights; rng = rng)
    weights_resampled = softmax(
        reduce(vcat, map(b -> weights[b:b, resampled_idxs[b, :]], 1:size(x)[end])),
        dims = 2,
    )

    # Works better with more samples
    z_prior, st_lux_ebm = m.sample_prior(m, m.IS_samples, ps, st_kan, st_lux, rng)
    return z_posterior,
        z_prior,
        st_lux_ebm,
        st_lux_gen,
        weights_resampled,
        resampled_idxs,
        noise
end

function marginal_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        weights_resampled,
        resampled_idxs,
        m,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
    )
    B, S = size(x)[end], size(z_posterior)[end]

    logprior_posterior, st_ebm =
        m.log_prior(z_posterior, m.prior, ps.ebm, st_kan.ebm, st_lux_ebm)
    logllhood, st_gen = log_likelihood_IS(
        z_posterior,
        x,
        m.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux_gen,
        noise;
        ε = m.ε,
    )

    marginal_llhood =
        loss_accum(weights_resampled, logprior_posterior, logllhood, resampled_idxs, B, S)

    logprior_prior, st_ebm =
        m.log_prior(z_prior, m.prior, ps.ebm, st_kan.ebm, st_ebm)
    ex_prior = m.prior.bool_config.contrastive_div ? mean(logprior_prior) : 0.0f0

    return -(marginal_llhood - ex_prior), st_ebm, st_gen
end

function closure(
        ps,
        z_posterior,
        z_prior,
        x,
        weights_resampled,
        resampled_idxs,
        m,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
    )
    return first(
        marginal_llhood(
            ps,
            z_posterior,
            z_prior,
            x,
            weights_resampled,
            resampled_idxs,
            m,
            st_kan,
            st_lux_ebm,
            st_lux_gen,
            noise,
        ),
    )
end

function grad_importance_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        weights_resampled,
        resampled_idxs,
        model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
    )

    return Enzyme.gradient(
        Enzyme.Reverse,
        Enzyme.Const(closure),
        ps,
        Enzyme.Const(z_posterior),
        Enzyme.Const(z_prior),
        Enzyme.Const(x),
        Enzyme.Const(weights_resampled),
        Enzyme.Const(resampled_idxs),
        Enzyme.Const(model),
        Enzyme.Const(st_kan),
        Enzyme.Const(st_lux_ebm),
        Enzyme.Const(st_lux_gen),
        Enzyme.Const(noise),
    )[1]
end

struct ImportanceLoss
    compiled_loss
    compiled_grad
end

function ImportanceLoss(
        ps::ComponentArray{T},
        st_kan::ComponentArray{T},
        st_lux::NamedTuple,
        model::T_KAM{T},
        x::AbstractArray{T};
        rng::AbstractRNG = Random.default_rng(),
    )::ImportanceLoss where {T <: Float32}
    z_posterior, z_prior, st_lux_ebm, st_lux_gen, weights_resampled, resampled_idxs, noise =
        sample_importance(ps, st_kan, Lux.testmode(st_lux), model, x; rng = rng)

    # Move all arguments to XLA device
    ps_xdev, z_posterior_xdev, z_prior_xdev, x_xdev = CUDA.@allowscalar (
        ps |> xdev, z_posterior |> xdev, z_prior |> xdev, x |> xdev,
    )
    weights_xdev, idxs_xdev, noise_xdev, st_kan_xdev = CUDA.@allowscalar (
        weights_resampled |> xdev, resampled_idxs |> xdev, noise |> xdev, st_kan |> xdev,
    )
    st_ebm_xdev = CUDA.@allowscalar Lux.trainmode(st_lux_ebm) |> xdev
    st_gen_xdev = CUDA.@allowscalar Lux.trainmode(st_lux_gen) |> xdev

    compiled_loss = Reactant.@compile marginal_llhood(
        ps_xdev,
        z_posterior_xdev,
        z_prior_xdev,
        x_xdev,
        weights_xdev,
        idxs_xdev,
        model,
        st_kan_xdev,
        st_ebm_xdev,
        st_gen_xdev,
        noise_xdev,
    )

    compiled_grad = Reactant.@compile grad_importance_llhood(
        ps_xdev,
        z_posterior_xdev,
        z_prior_xdev,
        x_xdev,
        weights_xdev,
        idxs_xdev,
        model,
        st_kan_xdev,
        st_ebm_xdev,
        st_gen_xdev,
        noise_xdev,
    )

    return ImportanceLoss(
        compiled_loss,
        compiled_grad
        # marginal_llhood,
        # grad_importance_llhood!
    )
end


function (l::ImportanceLoss)(
        ps::ComponentArray{T},
        ∇::ComponentArray{T},
        st_kan::ComponentArray{T},
        st_lux::NamedTuple,
        model::T_KAM{T},
        x::AbstractArray{T};
        train_idx::Int = 1,
        rng::AbstractRNG = Random.default_rng(),
    )::Tuple{T, AbstractArray{T}, NamedTuple, NamedTuple} where {T <: Float32}

    z_posterior, z_prior, st_lux_ebm, st_lux_gen, weights_resampled, resampled_idxs, noise =
        sample_importance(ps, st_kan, Lux.testmode(st_lux), model, x; rng = rng)

    # Move all arguments to XLA device once
    ps_xdev, z_posterior_xdev, z_prior_xdev, x_xdev = CUDA.@allowscalar (
        ps |> xdev, z_posterior |> xdev, z_prior |> xdev, x |> xdev,
    )
    weights_xdev, idxs_xdev, noise_xdev, st_kan_xdev = CUDA.@allowscalar (
        weights_resampled |> xdev, resampled_idxs |> xdev, noise |> xdev, st_kan |> xdev,
    )
    st_ebm_xdev = CUDA.@allowscalar Lux.trainmode(st_lux_ebm) |> xdev
    st_gen_xdev = CUDA.@allowscalar Lux.trainmode(st_lux_gen) |> xdev

    ∇ = l.compiled_grad(
        ps_xdev,
        z_posterior_xdev,
        z_prior_xdev,
        x_xdev,
        weights_xdev,
        idxs_xdev,
        model,
        st_kan_xdev,
        st_ebm_xdev,
        st_gen_xdev,
        noise_xdev,
    )

    all(iszero.(∇)) && error("All zero importance grad")
    any(isnan.(∇)) && error("NaN in importance grad")

    loss, st_lux_ebm, st_lux_gen = l.compiled_loss(
        ps_xdev,
        z_posterior_xdev,
        z_prior_xdev,
        x_xdev,
        weights_xdev,
        idxs_xdev,
        model,
        st_kan_xdev,
        st_ebm_xdev,
        st_gen_xdev,
        noise_xdev,
    )

    return loss, ∇, st_lux_ebm, st_lux_gen
end

end
