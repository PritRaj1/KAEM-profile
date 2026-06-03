module VariationalTraining

export VariationalLoss

using ComponentArrays, Enzyme, Statistics, Lux, Optimisers, NNlib

using ..Utils
using ..KAEM_model

include("../gen/loglikelihoods.jl")
include("../ebm/mixture_selection.jl")
using .LogLikelihoods: log_likelihood_MALA
using .MixtureChoice: choose_component

function elbo_loss(
        ps,
        x,
        ε,
        model,
        st_kan,
        st_lux_enc,
        st_lux_ebm,
        st_lux_gen,
        noise,
        β,
        component_mask,
    )
    z_posterior, log_q, μ, logvar, st_enc = model.encoder(
        ps.enc,
        st_lux_enc,
        x,
        ε;
        component_mask = component_mask,
    )

    log_p, st_ebm = model.log_prior(
        z_posterior,
        model.prior,
        ps.ebm,
        st_kan.ebm,
        st_lux_ebm,
        st_kan.quad;
        ula = false,
        component_mask = component_mask,
    )

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

    kl = log_q .- log_p

    reg, st_ebm, st_gen = model.kan_regularizer(
        z_posterior,
        model,
        ps,
        st_kan,
        st_ebm,
        st_gen
    )

    loss = reg - (mean(logllhood) - β * mean(kl))
    return (loss, st_ebm, st_gen, st_enc)
end

struct VariationalLoss
    model
    beta::AbstractArray{Float32}
end

function (l::VariationalLoss)(
        opt_state,
        ps,
        st_kan,
        st_lux,
        x,
        train_idx,
        st_rng,
    )
    β = l.beta[train_idx]

    ε = st_rng.encoder_noise
    noise = st_rng.train_noise
    Q, P, S = l.model.prior.q_size, l.model.prior.p_size, l.model.batch_size

    component_mask = (
        l.model.encoder.bool_config.mixture_model ?
            choose_component(ps.ebm.dist.α, S, Q, P, st_rng) :
            nothing
    )

    st_lux_enc = st_lux.enc
    st_lux_ebm = st_lux.ebm
    st_lux_gen = st_lux.gen

    dps = Enzyme.make_zero(ps)
    _, (loss, st_lux_ebm, st_lux_gen, st_lux_enc) = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        Const(elbo_loss),
        Active,
        Duplicated(ps, dps),
        Const(x),
        Const(ε),
        Const(l.model),
        Const(st_kan),
        Const(st_lux_enc),
        Const(st_lux_ebm),
        Const(st_lux_gen),
        Const(noise),
        Const(β),
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
