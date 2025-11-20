module ThermodynamicIntegration

export thermodynamic_loss

using ComponentArrays, Random, Enzyme, Statistics, Lux

using ..Utils
using ..KAEM_model

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_MALA

function sample_thermo(
        ps,
        st_kan,
        st_lux,
        model,
        x;
        train_idx = 1,
        rng = Random.default_rng(),
        swap_replica_idxs = nothing,
    )
    temps = collect(Float32, [(k / model.N_t)^model.p[train_idx] for k in 0:model.N_t])
    z, st_lux = model.posterior_sampler(
        model,
        ps,
        st_kan,
        st_lux,
        x;
        temps = temps[2:end],
        rng = rng,
        swap_replica_idxs = swap_replica_idxs,
    )

    Δt = temps[2:end] - temps[1:(end - 1)]
    tempered_noise = randn(rng, Float32, model.lkhood.x_shape..., prod(size(z)[3:4]))
    noise = randn(rng, Float32, model.lkhood.x_shape..., size(x)[end])
    return z, Δt, st_lux, noise, tempered_noise
end

function marginal_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        Δt,
        model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
        tempered_noise
    )
    Q, P, S, num_temps = size(z_posterior)
    log_ss = 0.0f0

    # Steppingstone estimator
    x_rep = model.lkhood.SEQ ? repeat(x, 1, 1, num_temps) : repeat(x, 1, 1, 1, num_temps)
    ll, st_gen = log_likelihood_MALA(
        reshape(z_posterior, Q, P, S * num_temps),
        x_rep,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux_gen,
        tempered_noise;
        ε = model.ε,
    )
    log_ss = sum(reshape(ll, num_temps, S) .* Δt) / S

    # MLE estimator
    logprior_pos, st_ebm = model.log_prior(
        z_posterior[:, :, :, num_temps - 1],
        model.prior,
        ps.ebm,
        st_kan.ebm,
        st_lux_ebm,
    )

    logprior, st_ebm =
        model.log_prior(z_prior, model.prior, ps.ebm, st_kan.ebm, st_ebm)
    ex_prior = model.prior.bool_config.contrastive_div ? mean(logprior) : 0.0f0

    logllhood, st_gen = log_likelihood_MALA(
        z_prior,
        x,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_gen,
        noise;
        ε = model.ε,
    )
    steppingstone_loss = mean(logllhood .* view(Δt, 1)) + log_ss
    return -(steppingstone_loss + mean(logprior_pos) - ex_prior),
        st_ebm,
        st_gen
end

function closure(
        ps,
        z_posterior,
        z_prior,
        x,
        Δt,
        model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
        tempered_noise
    )
    return first(
        marginal_llhood(
            ps,
            z_posterior,
            z_prior,
            x,
            Δt,
            model,
            st_kan,
            st_lux_ebm,
            st_lux_gen,
            noise,
            tempered_noise,
        ),
    )
end

function grad_thermo_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        Δt,
        model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
        tempered_noise
    )
    return first(
        Enzyme.gradient(
            Enzyme.Reverse,
            Enzyme.Const(closure),
            ps,
            Enzyme.Const(z_posterior),
            Enzyme.Const(z_prior),
            Enzyme.Const(x),
            Enzyme.Const(Δt),
            Enzyme.Const(model),
            Enzyme.Const(st_kan),
            Enzyme.Const(st_lux_ebm),
            Enzyme.Const(st_lux_gen),
            Enzyme.Const(noise),
            Enzyme.Const(tempered_noise)
        )
    )
end

function thermodynamic_loss(
        ps,
        st_kan,
        st_lux,
        model,
        x,
        train_idx,
        rng,
        swap_replica_idxs,
    )
    z_posterior, Δt, st_lux, noise, tempered_noise = sample_thermo(
        ps,
        st_kan,
        Lux.testmode(st_lux),
        model,
        x;
        train_idx = train_idx,
        rng = rng,
        swap_replica_idxs = swap_replica_idxs,
    )
    st_lux_ebm, st_lux_gen = st_lux.ebm, st_lux.gen
    z_prior, st_ebm =
        model.sample_prior(model, size(x)[end], ps, st_kan, Lux.testmode(st_lux), rng)

    ∇ = grad_thermo_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        Δt,
        model,
        st_kan,
        Lux.trainmode(st_lux_ebm),
        Lux.trainmode(st_lux_gen),
        noise,
        tempered_noise,
    )

    loss, st_lux_ebm, st_lux_gen = marginal_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        Δt,
        model,
        st_kan,
        Lux.trainmode(st_lux_ebm),
        Lux.trainmode(st_lux_gen),
        noise,
        tempered_noise,
    )
    return loss, ∇, st_lux_ebm, st_lux_gen
end

end
