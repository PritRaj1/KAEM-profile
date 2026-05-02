module LangevinSampling

export initialize_pCNL_sampler, initialize_ULA_sampler, LangevinSampler

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

include("ula_step.jl")
using .ULA_Step

include("../ebm/mixture_selection.jl")
using .MixtureChoice: choose_component

struct LangevinSampler
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

function _dims(model, prior_sampling_bool)
    Q, S = model.prior.q_size, model.batch_size
    P = model.prior.bool_config.mixture_model ? 1 : model.prior.p_size
    num_temps = (model.N_t > 1 && !prior_sampling_bool) ? model.N_t : 1
    return Q, P, S, num_temps, num_temps > 1
end

function _xchange(exchange_type, thermo_bool, Q, P, S, num_temps)
    return exchange_type != "none" && thermo_bool ?
        ReplicaXchange(Q, P, S, num_temps) : NoExchange()
end

function initialize_pCNL_sampler(
        model::KAEM{T};
        δ::T = 0.01f0,
        N::Int = 20,
        prior_sampling_bool::Bool = false,
        exchange_type::String = "none",
    ) where {T}

    @assert 0.0f0 < δ < 2.0f0 "pCNL δ must be in (0, 2), got $δ"
    Q, P, S, num_temps, thermo_bool = _dims(model, prior_sampling_bool)

    # pCNL coefficients: https://arxiv.org/abs/2408.14325 eq. 8
    denom = 2.0f0 + δ
    nc = sqrt(8.0f0 * δ) / denom
    z_c = (2.0f0 - δ) / denom
    inv_2σ2 = 1.0f0 / (2.0f0 * nc^2)

    log_dist = prior_sampling_bool ? unadjusted_logprior : unadjusted_logpos
    eval_dist = prior_sampling_bool ? per_sample_logprior : per_sample_logpos
    kernel = PcnlKernel(
        Q, P, S, num_temps, z_c, nc, inv_2σ2, log_dist, eval_dist,
        _xchange(exchange_type, thermo_bool, Q, P, S, num_temps),
    )
    return LangevinSampler(
        prior_sampling_bool, N, model, Q, P, S, num_temps, thermo_bool, kernel,
    )
end

function initialize_ULA_sampler(
        model::KAEM{T};
        η::T = 1.0f-3,
        N::Int = 20,
        prior_sampling_bool::Bool = false,
        exchange_type::String = "none",
    ) where {T}

    Q, P, S, num_temps, thermo_bool = _dims(model, prior_sampling_bool)
    log_dist = prior_sampling_bool ? unadjusted_logprior : unadjusted_logpos
    kernel = UlaKernel(
        η, sqrt(2 * η), log_dist,
        _xchange(exchange_type, thermo_bool, Q, P, S, num_temps),
    )
    return LangevinSampler(
        prior_sampling_bool, N, model, Q, P, S, num_temps, thermo_bool, kernel,
    )
end

function (sampler::LangevinSampler)(
        ps,
        st_kan,
        st_lux,
        x,
        st_rng;
        temps = [1.0f0],
    )
    """Langevin posterior sampler. Returns z ~ p(z|x) (or prior if prior_sampling_bool)."""
    model = sampler.model
    Q, P, S, num_temps = sampler.Q, sampler.P, sampler.S, sampler.num_temps
    prior_sampling_bool = sampler.prior_sampling_bool

    z_flat = if model.prior.bool_config.ula && prior_sampling_bool
        rv = st_rng.posterior_its
        model.prior.prior_type == "lognormal" ? exp.(rv) : rv
    else
        model_copy = model
        @reset model_copy.batch_size = S * num_temps
        @reset model_copy.prior.s_size = S * num_temps
        for i in 1:model_copy.prior.depth
            @reset model_copy.prior.fcns_qp[i].basis_function.S = S * num_temps
        end
        its = model.prior.bool_config.mixture_model ?
            MixITSSampler(model_copy.prior) : UnivITSSampler(model_copy.prior)
        first(its(ps, st_kan, st_lux, st_rng; ula_init = true))
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

    x_t = !prior_sampling_bool ? (
            model.lkhood.SEQ ? repeat(x, 1, 1, num_temps) :
            (model.use_pca ? repeat(x, 1, num_temps) : repeat(x, 1, 1, 1, num_temps))
        ) : x

    noise = st_rng.mcmc_noise
    log_u_mh = st_rng.log_mh
    log_u_swap = st_rng.log_swap
    mask_swap_1 = num_temps > 1 ? st_rng.swap_mask_1 : nothing
    mask_swap_2 = num_temps > 1 ? st_rng.swap_mask_2 : nothing

    N_steps = sampler.N
    kernel = sampler.kernel
    state = (1, z_flat)
    @trace while first(state) <= N_steps
        i, z_acc = state
        z_new = kernel(
            i,
            z_acc,
            x_t,
            temps_gpu,
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
            temps
        )
        state = (i + 1, z_new)
    end

    z = reshape(last(state), Q, P, S, num_temps)
    if prior_sampling_bool
        st_lux = st_lux.ebm
        z = dropdims(z; dims = 4)
    end

    return z, st_lux
end

end
