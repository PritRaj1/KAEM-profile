module pCNL_Step

export PcnlKernel

using ..Utils
using ..KAEM_model

include("updates.jl")
using .LangevinUpdates

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_MALA

struct PcnlKernel
    Q::Int
    P::Int
    S::Int
    num_temps::Int
    log_dist::Function
    eval_dist::Function
end

function (k::PcnlKernel)(
        i,
        z_i,
        accept_count,
        x_t,
        temps_gpu,
        z_c,
        n_c,
        inv_2σ2,
        model,
        lkhood_copy,
        ps,
        st_kan,
        st_lux,
        noise,
        log_u_mh,
        log_u_swap,
        mask_swap_1,
        mask_swap_2,
        component_mask,
        shift_down,
        shift_up,
        temps,
    )
    Q, P, S, num_temps = k.Q, k.P, k.S, k.num_temps

    ξ = noise[:, :, :, i]
    zero_vector = zero(x_t)

    # pCNL proposal + MH: https://arxiv.org/abs/2408.14325 eq. 8-9
    # Dℓ(u) = ∇log_target(u) + u
    ∇z = unadjusted_grad(
        z_i, x_t, temps_gpu, model, ps, st_kan, st_lux,
        component_mask, k.log_dist,
    )
    Dell_old = ∇z .+ z_i

    # Proposal
    m_old = z_c .* z_i .+ (1.0f0 .- z_c) .* Dell_old
    z_prop = m_old .+ n_c .* ξ

    # Reverse Dℓ(v)
    ∇z_prop = unadjusted_grad(
        z_prop, x_t, temps_gpu, model, ps, st_kan, st_lux,
        component_mask, k.log_dist,
    )
    Dell_new = ∇z_prop .+ z_prop
    m_new = z_c .* z_prop .+ (1.0f0 .- z_c) .* Dell_new

    # Per-sample log-target
    logp_old = k.eval_dist(
        z_i, x_t, temps_gpu, model, ps, st_kan, st_lux,
        component_mask, zero_vector,
    )
    logp_new = k.eval_dist(
        z_prop, x_t, temps_gpu, model, ps, st_kan, st_lux,
        component_mask, zero_vector,
    )

    # MH: log α = π(v)/π(u) · q(u|v)/q(v|u)
    fwd_sq = dropdims(sum((z_prop .- m_old) .^ 2; dims = (1, 2)); dims = (1, 2))
    bwd_sq = dropdims(sum((z_i .- m_new) .^ 2; dims = (1, 2)); dims = (1, 2))
    log_alpha = logp_new .- logp_old .+ inv_2σ2 .* (fwd_sq .- bwd_sq)

    # Accept/reject
    log_u = log_u_mh[:, i]
    accept = max.(sign.(log_alpha .- log_u), 0.0f0)
    accept_z = reshape(accept, 1, 1, S * num_temps)
    z_mh = accept_z .* z_prop .+ (1.0f0 .- accept_z) .* z_i

    # Per-temperature accept counts
    accept_per_temp = dropdims(
        sum(reshape(accept, num_temps, S); dims = 2); dims = 2,
    )

    # Replica exchange
    z_xch = model.xchange_func(
        i,
        z_mh,
        x_t,
        temps,
        model,
        lkhood_copy,
        ps,
        st_kan,
        st_lux,
        log_u_swap,
        mask_swap_1,
        mask_swap_2,
        shift_down,
        shift_up,
    )

    return z_xch, accept_count .+ accept_per_temp
end

end
