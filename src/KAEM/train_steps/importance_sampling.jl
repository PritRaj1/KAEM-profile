module ImportanceSampling

using ComponentArrays, Enzyme, Statistics, Lux, Optimisers
using NNlib: softmax

export ImportanceLoss

using ..Utils
using ..KAEM_model

include("../gen/loglikelihoods.jl")
include("../ebm/mixture_selection.jl")
using .LogLikelihoods: log_likelihood_IS
using .MixtureChoice: choose_component

function loss_accum(
        logprior,
        logllhood,
        resampled_mask,
        B,
        N,
    )
    sel_prior = sum(reshape(logprior, 1, 1, N) .* resampled_mask; dims = 3)
    is_prior = mean(sel_prior; dims = 2)

    ll = PermutedDimsArray(view(logllhood, :, :, :), (1, 3, 2))
    sel_ll = sum(ll .* resampled_mask; dims = 3)
    is_ll = mean(sel_ll; dims = 2)

    return mean(is_prior + is_ll)
end

function sample_importance(
        ps,
        st_kan,
        st_lux,
        m,
        x,
        st_rng
    )
    # Prior is proposal for importance sampling
    z_posterior, st_lux_ebm = m.sample_prior(ps, st_kan, st_lux, st_rng)
    noise = st_rng.train_noise
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
    N = m.batch_size
    weights = softmax(logllhood, dims = 2)
    resampled_indices = m.lkhood.resample_z(weights, st_rng)
    resampled_mask = resampled_indices .== reshape(1:N, 1, 1, N) |> Lux.f32

    # Works better with more samples
    z_prior, st_lux_ebm = m.sample_prior(ps, st_kan, st_lux, st_rng)

    # Get component mask for mixture model normalization
    Q, P, S = m.prior.q_size, m.prior.p_size, m.batch_size
    component_mask = (
        m.prior.bool_config.mixture_model && !m.prior.bool_config.contrastive_div ?
            choose_component(ps.ebm.dist.α, S, Q, P, st_rng) :
            nothing
    )

    return z_posterior,
        z_prior,
        st_lux_ebm,
        st_lux_gen,
        resampled_mask,
        noise,
        component_mask
end

function marginal_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        resampled_mask,
        m,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
        component_mask,
    )
    B, N = m.batch_size, m.batch_size

    logprior_posterior, st_ebm = m.log_prior(
        z_posterior,
        m.prior,
        ps.ebm,
        st_kan.ebm,
        st_lux_ebm,
        st_kan.quad;
        component_mask = component_mask
    )
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

    logprior_prior, st_ebm = m.log_prior(
        z_prior,
        m.prior,
        ps.ebm,
        st_kan.ebm,
        st_ebm,
        st_kan.quad;
        component_mask = component_mask
    )
    ex_prior = m.prior.bool_config.contrastive_div ? mean(logprior_prior) : 0.0f0

    marginal_llhood_val = loss_accum(
        logprior_posterior,
        logllhood,
        resampled_mask,
        B,
        N
    )

    reg, st_ebm, st_gen = m.kan_regularizer(
        z_posterior,
        m,
        ps,
        st_kan,
        st_ebm,
        st_gen
    )

    loss = reg - (marginal_llhood_val - ex_prior)
    return (loss, st_ebm, st_gen)
end

struct ImportanceLoss
    model
end

function (l::ImportanceLoss)(
        opt_state,
        ps,
        st_kan,
        st_lux,
        x,
        train_idx,
        st_rng,
    )

    z_posterior, z_prior, st_ebm, st_gen, resampled_mask, noise, component_mask =
        sample_importance(ps, st_kan, st_lux, l.model, x, st_rng)

    dps = Enzyme.make_zero(ps)
    _, (loss, st_lux_ebm, st_lux_gen) = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        Const(marginal_llhood),
        Active,
        Duplicated(ps, dps),
        Const(z_posterior),
        Const(z_prior),
        Const(x),
        Const(resampled_mask),
        Const(l.model),
        Const(st_kan),
        Const(st_ebm),
        Const(st_gen),
        Const(noise),
        Const(component_mask),
    )

    opt_state_gen, ps_gen_new = Optimisers.update(opt_state.gen, ps.gen, dps.gen)
    opt_state_ebm, ps_ebm_new = Optimisers.update(opt_state.ebm, ps.ebm, dps.ebm)
    opt_state_enc, ps_enc_new = Optimisers.update(opt_state.enc, ps.enc, dps.enc)
    @views ps[:gen] .= ps_gen_new
    @views ps[:ebm] .= ps_ebm_new
    @views ps[:enc] .= ps_enc_new
    opt_state = (gen = opt_state_gen, ebm = opt_state_ebm, enc = opt_state_enc)
    return loss, ps, opt_state, st_lux_ebm, st_lux_gen
end

end
