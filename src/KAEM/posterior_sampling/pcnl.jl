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
    β
    coeff_z # √(1 - β²) + β²
    β2 # β²
    model
    Q
    P
    S
    num_temps
    thermo_bool
    log_dist # summed log-posterior (for gradient)
    eval_dist # per-sample log-posterior (for MH)
end

function initialize_pCNL_sampler(
        model::KAEM{T};
        β::T = 0.1f0,
        N::Int = 20,
    ) where {T}

    Q, S = model.prior.q_size, model.batch_size
    P = model.prior.bool_config.mixture_model ? 1 : model.prior.p_size
    num_temps = model.N_t > 1 ? model.N_t : 1
    thermo_bool = num_temps > 1

    return pCNL_sampler(
        N,
        β,
        sqrt(1.0f0 - β^2) + β^2,
        β^2,
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

    # Gradient of log-posterior (same kernel as ULA)
    ∇z = unadjusted_grad(
        z_i,
        x_t,
        temps_gpu,
        model,
        ps,
        st_kan,
        st_lux,
        component_mask,
        sampler.log_dist,
    )

    # pCNL proposal: z' = (√(1-β²) + β²)·z + β²·∇log_pos + β·ξ
    z_prop = sampler.coeff_z .* z_i .+ sampler.β2 .* ∇z .+ sampler.β .* ξ

    # MH accept/reject (Metropolis ratio, HLO-compatible)
    logpos_old = sampler.eval_dist(
        z_i, x_t, temps_gpu, model, ps, st_kan, st_lux, component_mask, zero_vector,
    )
    logpos_new = sampler.eval_dist(
        z_prop, x_t, temps_gpu, model, ps, st_kan, st_lux, component_mask, zero_vector,
    )

    log_alpha = logpos_new .- logpos_old
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
