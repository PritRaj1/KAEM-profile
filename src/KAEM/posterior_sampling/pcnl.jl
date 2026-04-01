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

include("../ebm/mixture_selection.jl")
using .MixtureChoice: choose_component

struct pCNL_sampler
    N
    δ
    z_coeff     # (2 - δ) / (2 + δ)
    grad_coeff  # 2δ / (2 + δ)
    noise_coeff # √(8δ) / (2 + δ)
    inv_2σ2     # (2 + δ)² / (2 · 8δ)
    model
    Q
    P
    S
    num_temps
    thermo_bool
    log_dist
    eval_dist
end

function initialize_pCNL_sampler(
        model::KAEM{T};
        δ::T = 0.01f0,
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

    # Dℓ(u) = ∇log_posterior(u) + u  (gradient w.r.t. Gaussian measure)
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

    # Per-sample log-posterior for MH ratio
    logpos_old = sampler.eval_dist(
        z_i, x_t, temps_gpu, model, ps, st_kan, st_lux, component_mask, zero_vector,
    )
    logpos_new = sampler.eval_dist(
        z_prop, x_t, temps_gpu, model, ps, st_kan, st_lux, component_mask, zero_vector,
    )

    # MH: log α = [π(v)/π(u)] · [q(u|v)/q(v|u)]
    # q(v|u) = N(v; m(u), σ²I), so log q ∝ -‖·‖²/(2σ²)
    fwd_sq = dropdims(sum((z_prop .- m_old) .^ 2; dims = (1, 2)); dims = (1, 2))
    bwd_sq = dropdims(sum((z_i .- m_new) .^ 2; dims = (1, 2)); dims = (1, 2))
    log_alpha = logpos_new .- logpos_old .+ sampler.inv_2σ2 .* (fwd_sq .- bwd_sq)

    # Accept/reject (HLO-compatible)
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
    """pCNL posterior sampler (Pezzetti 2024). Returns z ~ p(z|x)."""
    model = sampler.model
    Q, P, S, num_temps = sampler.Q, sampler.P, sampler.S, sampler.num_temps

    # Initialize from prior
    z_flat = begin
        model_copy = model

        @reset model_copy.batch_size = S * num_temps
        @reset model_copy.prior.s_size = S * num_temps
        for i in 1:model_copy.prior.depth
            @reset model_copy.prior.fcns_qp[i].basis_function.S = S * num_temps
        end

        z_init, _ = begin
            if model.prior.bool_config.mixture_model
                sample_mixture(
                    model_copy.prior,
                    ps.ebm,
                    st_kan.ebm,
                    st_lux.ebm,
                    st_kan.quad,
                    st_rng;
                    ula_init = true,
                )
            else
                sample_univariate(
                    model_copy.prior,
                    ps.ebm,
                    st_kan.ebm,
                    st_lux.ebm,
                    st_kan.quad,
                    st_rng;
                    ula_init = true,
                )
            end
        end

        z_init
    end

    lkhood_copy = model.lkhood

    for i in 1:model.prior.depth
        @reset model.prior.fcns_qp[i].basis_function.S = S * num_temps
    end

    if !model.lkhood.SEQ && !model.lkhood.CNN
        for i in 1:model.lkhood.generator.depth
            @reset model.lkhood.generator.Φ_fcns[i].basis_function.S = S * num_temps
        end
    end

    # Mask used for mixture sampling
    component_mask = (
        model.prior.bool_config.mixture_model && !model.prior.bool_config.contrastive_div ?
            choose_component(ps.ebm.dist.α, S, Q, P, st_rng) :
            nothing
    )
    component_mask = isnothing(component_mask) ? nothing : repeat(component_mask, 1, 1, num_temps)

    @reset model.prior.s_size = S * num_temps
    @reset model.lkhood.generator.s_size = S * num_temps
    temps_gpu = repeat(temps, S)

    N_steps = sampler.N
    x_t = model.lkhood.SEQ ? repeat(x, 1, 1, num_temps) :
        (model.use_pca ? repeat(x, 1, num_temps) : repeat(x, 1, 1, 1, num_temps))

    # Pre-allocate noise
    noise = st_rng.ula_noise
    log_u_mh = st_rng.log_mh
    log_u_swap = st_rng.log_swap

    # DEO masks + shift matrices
    mask_swap_1 = num_temps > 1 ? st_rng.swap_mask_1 : nothing
    mask_swap_2 = num_temps > 1 ? st_rng.swap_mask_2 : nothing
    shift_down = num_temps > 1 ? st_rng.shift_down : nothing
    shift_up = num_temps > 1 ? st_rng.shift_up : nothing

    state = (1, z_flat)
    @trace while first(state) <= N_steps
        i, z_acc = state
        z_new = step(
            i,
            z_acc,
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
        state = (i + 1, z_new)
    end

    z = reshape(last(state), Q, P, S, num_temps)

    return z, st_lux
end

end
