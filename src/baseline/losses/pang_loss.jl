module PangLoss

export PangTrainStep

using Enzyme, Optimisers, Lux, Statistics, ComponentArrays

using ..PangEBMSampling: langevin_prior, langevin_posterior

function pang_total_loss(ps, x, z_prior, z_post, model, st, α_cd)
    E_post, _ = model.energy_net(z_post, ps.ebm, st.ebm)
    E_prior, _ = model.energy_net(z_prior, ps.ebm, st.ebm)
    cd_loss = mean(E_post) - mean(E_prior)

    x_recon, _ = model.generator(z_post, ps.gen, st.gen)
    batch_size = Float32(size(x, ndims(x)))
    recon_loss = sum((x_recon .- x) .^ 2) / batch_size

    loss = recon_loss + α_cd * cd_loss
    return (loss, st)
end

struct PangTrainStep
    model
    α_cd
end

function (l::PangTrainStep)(opt_state_gen, opt_state_ebm, ps, st, x, st_rng)
    z_prior = langevin_prior(l.model, ps, st, st_rng)
    z_post = langevin_posterior(l.model, x, ps, st, st_rng)

    dps = Enzyme.make_zero(ps)
    _, (loss, st_new) = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        Const(pang_total_loss),
        Active,
        Duplicated(ps, dps),
        Const(x),
        Const(z_prior),
        Const(z_post),
        Const(l.model),
        Const(Lux.trainmode(st)),
        Const(l.α_cd),
    )

    opt_state_gen, ps_gen_new = Optimisers.update(opt_state_gen, ps.gen, dps.gen)
    opt_state_ebm, ps_ebm_new = Optimisers.update(opt_state_ebm, ps.ebm, dps.ebm)
    @views ps[:gen] .= ps_gen_new
    @views ps[:ebm] .= ps_ebm_new
    return loss, ps, opt_state_gen, opt_state_ebm, st_new
end

end
