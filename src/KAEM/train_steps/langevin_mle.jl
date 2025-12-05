module LangevinMLE

export LangevinLoss

using ComponentArrays, Random, Enzyme, Statistics, Lux, Optimisers
using MLUtils: randn_like

using ..Utils
using ..KAEM_model

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_MALA

function sample_langevin(
        ps,
        st_kan,
        st_lux,
        model,
        x;
        rng = Random.MersenneTwister(1),
    )
    z, st_lux, = model.posterior_sampler(ps, st_kan, st_lux, x; rng = rng)
    noise = randn_like(Lux.replicate(rng), x)
    return z[:, :, :, 1], st_lux, noise
end

function marginal_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise
    )

    logprior_pos, st_ebm =
        model.log_prior(z_posterior, model.prior, ps.ebm, st_kan.ebm, st_lux_ebm)
    logllhood, st_gen = log_likelihood_MALA(
        z_posterior,
        x,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux_gen,
        noise;
        ε = model.ε,
    )

    logprior, st_ebm =
        model.log_prior(z_prior, model.prior, ps.ebm, st_kan.ebm, st_ebm)
    ex_prior = model.prior.bool_config.contrastive_div ? mean(logprior) : 0.0f0
    return -(mean(logprior_pos) + mean(logllhood) - ex_prior),
        st_ebm,
        st_gen
end

function closure(
        ps,
        z_posterior,
        z_prior,
        x,
        model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise
    )
    return first(
        marginal_llhood(
            ps,
            z_posterior,
            z_prior,
            x,
            model,
            st_kan,
            st_lux_ebm,
            st_lux_gen,
            noise,
        ),
    )
end

function grad_langevin_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise
    )
    return first(
        Enzyme.gradient(
            Enzyme.Reverse,
            Enzyme.Const(closure),
            ps,
            Enzyme.Const(z_posterior),
            Enzyme.Const(z_prior),
            Enzyme.Const(x),
            Enzyme.Const(model),
            Enzyme.Const(st_kan),
            Enzyme.Const(st_lux_ebm),
            Enzyme.Const(st_lux_gen),
            Enzyme.Const(noise)
        )
    )
end

struct LangevinLoss
    model
end

function (l::LangevinLoss)(
        opt_state,
        ps,
        st_kan,
        st_lux,
        x,
        train_idx,
        rng,
        swap_replica_idxs,
    )
    z_posterior, st_new, noise =
        sample_langevin(ps, st_kan, Lux.trainmode(st_lux), l.model, x; rng = rng)
    st_lux_ebm, st_lux_gen = st_new.ebm, st_new.gen
    z_prior, st_lux_ebm =
        l.model.sample_prior(l.model, ps, st_kan, st_lux, rng)

    ∇ = grad_langevin_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        l.model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
    )

    loss, st_lux_ebm, st_lux_gen = marginal_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        l.model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
    )

    opt_state, ps = Optimisers.update(opt_state, ps, ∇)
    return loss, ps, opt_state, st_lux_ebm, st_lux_gen
end

end
