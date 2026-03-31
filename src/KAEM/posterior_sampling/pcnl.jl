module pCNL_sampling

export initialize_pCNL_sampler, pCNL_sampler

using Reactant: @trace
using LinearAlgebra,
    Lux,
    Accessors,
    Statistics,
    ComponentArrays

using ..Utils
using ..KAEM_model
using ..KAEM_model.InverseTransformSampling

include("updates.jl")
using .LangevinUpdates

include("sampler_init.jl")
using .SamplerInit

struct pCNL_sampler
    N
    δ
    z_coeff     # (2 - δ) / (2 + δ)
    grad_coeff  # 2δ / (2 + δ)
    noise_coeff # √(8δ) / (2 + δ)
    inv_2σ2     # 1 / (2 · noise_coeff²)
    model
    Q
    P
    S
    num_temps
    thermo_bool
    log_dist    # summed log-posterior (for gradient)
    eval_dist   # per-sample log-posterior (for MH)
end

function initialize_pCNL_sampler(
        model::KAEM{T};
        δ::T = 0.5f0,
        N::Int = 20,
    ) where {T}

    @assert 0.0f0 < δ < 2.0f0 "pCNL step size δ must be in (0, 2), got $δ"

    Q, S = model.prior.q_size, model.batch_size
    P = model.prior.bool_config.mixture_model ? 1 : model.prior.p_size
    num_temps = model.N_t > 1 ? model.N_t : 1
    thermo_bool = num_temps > 1

    denom = 2.0f0 + δ
    nc = sqrt(8.0f0 * δ) / denom

    return pCNL_sampler(
        N,
        δ,
        (2.0f0 - δ) / denom,
        2.0f0 * δ / denom,
        nc,
        1.0f0 / (2.0f0 * nc^2),
        model,
        Q,
        P,
        S,
        num_temps,
        thermo_bool,
        unadjusted_logpos,
        per_sample_logpos,
    )
end

function step(
        i,
        z_i,
        x_t,
        temps,
        temps_gpu,
        sampler,
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
    )

    ξ = noise[:, :, :, i]
    zero_vector = zero(x_t)

    # Dℓ(u) = ∇log_posterior(u) + u  (gradient w.r.t. Gaussian reference)
    ∇z = unadjusted_grad(
        z_i, x_t, temps_gpu, model, ps, st_kan, st_lux,
        component_mask, sampler.log_dist,
    )
    Dell_old = ∇z .+ z_i

    # pCNL proposal (Pezzetti 2024 eq. 8)
    m_old = sampler.z_coeff .* z_i .+ sampler.grad_coeff .* Dell_old
    z_prop = m_old .+ sampler.noise_coeff .* ξ

    # Dℓ(v) for reverse proposal mean
    ∇z_prop = unadjusted_grad(
        z_prop, x_t, temps_gpu, model, ps, st_kan, st_lux,
        component_mask, sampler.log_dist,
    )
    Dell_new = ∇z_prop .+ z_prop
    m_new = sampler.z_coeff .* z_prop .+ sampler.grad_coeff .* Dell_new

    # Per-sample log-posterior
    logpos_old = sampler.eval_dist(
        z_i, x_t, temps_gpu, model, ps, st_kan, st_lux, component_mask, zero_vector,
    )
    logpos_new = sampler.eval_dist(
        z_prop, x_t, temps_gpu, model, ps, st_kan, st_lux, component_mask, zero_vector,
    )

    # Proposal density correction: log q(u|v) - log q(v|u)
    fwd_sq = dropdims(sum((z_prop .- m_old) .^ 2; dims = (1, 2)); dims = (1, 2))
    bwd_sq = dropdims(sum((z_i .- m_new) .^ 2; dims = (1, 2)); dims = (1, 2))
    log_proposal_ratio = sampler.inv_2σ2 .* (fwd_sq .- bwd_sq)

    # MH accept/reject (HLO-compatible)
    log_alpha = logpos_new .- logpos_old .+ log_proposal_ratio
    log_u = log_u_mh[:, i]
    accept = max.(sign.(log_alpha .- log_u), 0.0f0)
    accept_z = reshape(accept, 1, 1, :)
    z_mh = accept_z .* z_prop .+ (1.0f0 .- accept_z) .* z_i

    # Replica exchange
    return model.xchange_func(
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
end

function (sampler::pCNL_sampler)(
        ps,
        st_kan,
        st_lux,
        x,
        st_rng;
        temps = [1.0f0],
    )
    """pCNL posterior sampler. Returns z ~ p(z|x)."""
    Q, P, S, num_temps = sampler.Q, sampler.P, sampler.S, sampler.num_temps

    ss = init_sampler_state(
        sampler.model, ps, st_kan, st_lux, st_rng, x,
        Q, P, S, num_temps,
    )
    model = ss.model
    temps_gpu = repeat(temps, S)
    log_u_mh = st_rng.log_mh

    state = (1, ss.z_flat)
    @trace while first(state) <= sampler.N
        i, z_acc = state
        z_new = step(
            i,
            z_acc,
            ss.x_t,
            temps,
            temps_gpu,
            sampler,
            model,
            ss.lkhood_copy,
            ps,
            st_kan,
            st_lux,
            ss.noise,
            log_u_mh,
            ss.log_u_swap,
            ss.mask_swap_1,
            ss.mask_swap_2,
            ss.component_mask,
            ss.shift_down,
            ss.shift_up,
        )
        state = (i + 1, z_new)
    end

    z = reshape(last(state), Q, P, S, num_temps)

    return z, st_lux
end

end
