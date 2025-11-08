module LangevinMLE

export langevin_loss

using ComponentArrays, Random, Enzyme, Statistics, Lux

using ..Utils
using ..T_KAM_model

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_MALA

function sample_langevin(
        ps,
        st_kan,
        st_lux,
        model,
        x;
        rng = Random.default_rng(),
    )
    z, st_lux, = model.posterior_sampler(model, ps, st_kan, st_lux, x; rng = rng)
    z = z[:, :, :, 1]
    noise = randn(rng, Float32, model.lkhood.x_shape..., size(z)[end]) |> pu
    return z, st_lux, noise
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
    )::Tuple{T, NamedTuple, NamedTuple} where {T <: Float32}

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


function langevin_loss(
        ps,
        st_kan,
        st_lux,
        model,
        x;
        train_idx = 1,
        rng = Random.default_rng(),
    )
    z_posterior, st_new, noise =
        sample_langevin(ps, st_kan, Lux.testmode(st_lux), model, x; rng = rng)
    st_lux_ebm, st_lux_gen = st_new.ebm, st_new.gen
    z_prior, st_lux_ebm =
        model.sample_prior(model, size(x)[end], ps, st_kan, Lux.testmode(st_lux), rng)

    ∇ = grad_langevin_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        model,
        st_kan,
        Lux.trainmode(st_lux_ebm),
        Lux.trainmode(st_lux_gen),
        noise,
    )

    all(iszero.(∇)) && error("All zero Langevin grad")
    any(isnan.(∇)) && error("NaN in Langevin grad")

    loss, st_lux_ebm, st_lux_gen = marginal_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        model,
        st_kan,
        Lux.trainmode(st_lux_ebm),
        Lux.trainmode(st_lux_gen),
        noise,
    )
    return loss, ∇, st_lux_ebm, st_lux_gen
end

end
