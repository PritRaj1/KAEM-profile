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
using ..KAEM_model.PopulationXchange

include("updates.jl")
using .LangevinUpdates

include("pcnl_step.jl")
using .pCNL_Step

include("../ebm/mixture_selection.jl")
using .MixtureChoice: choose_component

struct pCNL_sampler
    prior_sampling_bool
    N
    model
    Q
    P
    S
    num_temps
    thermo_bool
    kernel
end

function initialize_pCNL_sampler(
        model::KAEM{T};
        δ::T = 0.01f0,
        N::Int = 20,
        prior_sampling_bool::Bool = false,
        exchange_type::String = "none",
    ) where {T}

    @assert 0.0f0 < δ < 2.0f0 "pCNL δ must be in (0, 2), got $δ"

    Q, S = model.prior.q_size, model.batch_size
    P = model.prior.bool_config.mixture_model ? 1 : model.prior.p_size
    num_temps = (model.N_t > 1 && !prior_sampling_bool) ? model.N_t : 1
    thermo_bool = num_temps > 1

    log_dist = prior_sampling_bool ? unadjusted_logprior : unadjusted_logpos
    eval_dist = prior_sampling_bool ? per_sample_logprior : per_sample_logpos
    xchange = (
        exchange_type != "none" && thermo_bool ?
            ReplicaXchange(Q, P, S, num_temps) : NoExchange()
    )
    kernel = PcnlKernel(Q, P, S, num_temps, log_dist, eval_dist, xchange)

    return pCNL_sampler(
        prior_sampling_bool, N, model, Q, P, S, num_temps, thermo_bool, kernel,
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
    """pCNL sampler (https://arxiv.org/abs/2408.14325)."""
    model = sampler.model
    Q, P, S, num_temps = sampler.Q, sampler.P, sampler.S, sampler.num_temps
    prior_sampling_bool = sampler.prior_sampling_bool

    # Initialize from prior
    z_flat = begin
        if model.prior.bool_config.ula && prior_sampling_bool
            rv = st_rng.posterior_its
            model.prior.prior_type == "lognormal" ? exp.(rv) : rv
        else
            model_copy = model
            @reset model_copy.batch_size = S * num_temps
            @reset model_copy.prior.s_size = S * num_temps
            for i in 1:model_copy.prior.depth
                @reset model_copy.prior.fcns_qp[i].basis_function.S = S * num_temps
            end

            z_init, _ = begin
                if model.prior.bool_config.mixture_model
                    sample_mixture(
                        model_copy.prior, ps.ebm, st_kan.ebm, st_lux.ebm,
                        st_kan.quad, st_rng; ula_init = true,
                    )
                else
                    sample_univariate(
                        model_copy.prior, ps.ebm, st_kan.ebm, st_lux.ebm,
                        st_kan.quad, st_rng; ula_init = true,
                    )
                end
            end
            z_init
        end
    end

    for i in 1:model.prior.depth
        @reset model.prior.fcns_qp[i].basis_function.S = S * num_temps
    end
    if !model.lkhood.SEQ && !model.lkhood.CNN
        for i in 1:model.lkhood.generator.depth
            @reset model.lkhood.generator.Φ_fcns[i].basis_function.S = S * num_temps
        end
    end

    component_mask = (
        model.prior.bool_config.mixture_model && !model.prior.bool_config.contrastive_div ?
            choose_component(ps.ebm.dist.α, S, Q, P, st_rng) :
            nothing
    )
    component_mask = isnothing(component_mask) ? nothing : repeat(component_mask, 1, 1, num_temps)

    @reset model.prior.s_size = S * num_temps
    @reset model.lkhood.generator.s_size = S * num_temps
    temps_gpu = repeat(temps, S)

    # Per-temperature coefficients: https://arxiv.org/abs/2408.14325 eq. 8
    δ_gpu = repeat(st_lux.delta, S)
    denom = 2.0f0 .+ δ_gpu
    nc = (8.0f0 .* δ_gpu) .^ 0.5f0 ./ denom
    z_c = reshape((2.0f0 .- δ_gpu) ./ denom, 1, 1, S * num_temps) .* 1.0f0
    n_c = reshape(nc, 1, 1, S * num_temps) .* 1.0f0
    inv_2σ2 = 1.0f0 ./ (2.0f0 .* nc .^ 2)

    N_steps = sampler.N
    x_t = !prior_sampling_bool ? (
            model.lkhood.SEQ ? repeat(x, 1, 1, num_temps) :
            (model.use_pca ? repeat(x, 1, num_temps) : repeat(x, 1, 1, 1, num_temps))
        ) : x

    noise = st_rng.mcmc_noise
    log_u_mh = st_rng.log_mh
    log_u_swap = st_rng.log_swap
    mask_swap_1 = num_temps > 1 ? st_rng.swap_mask_1 : nothing
    mask_swap_2 = num_temps > 1 ? st_rng.swap_mask_2 : nothing
    kernel = sampler.kernel
    accept_count = zero(st_lux.delta)

    state = (1, z_flat, accept_count)
    @trace while first(state) <= N_steps
        i, z_acc, ac = state
        z_new, ac_new = kernel(
            i,
            z_acc,
            ac,
            x_t,
            temps_gpu,
            z_c,
            n_c,
            inv_2σ2,
            model,
            ps,
            st_kan,
            st_lux,
            noise,
            log_u_mh,
            log_u_swap,
            mask_swap_1,
            mask_swap_2,
            component_mask,
            temps,
        )
        state = (i + 1, z_new, ac_new)
    end

    _, z_final, final_accept = state
    z = reshape(z_final, Q, P, S, num_temps)
    accept_rate = final_accept ./ (N_steps * S)

    if prior_sampling_bool
        st_lux = st_lux.ebm
        z = dropdims(z; dims = 4)
    end

    return z, st_lux, accept_rate
end

end
