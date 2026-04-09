module ULA_sampling

export initialize_ULA_sampler, ULA_sampler

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

include("ula_step.jl")
using .ULA_Step

include("../ebm/mixture_selection.jl")
using .MixtureChoice: choose_component

struct ULA_sampler
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

function initialize_ULA_sampler(
        model::KAEM{T};
        η::T = 1.0f-3,
        prior_sampling_bool::Bool = false,
        N::Int = 20,
        exchange_type::String = "none",
    ) where {T}

    Q, S = model.prior.q_size, model.batch_size
    P = model.prior.bool_config.mixture_model ? 1 : model.prior.p_size
    num_temps = (model.N_t > 1 && !prior_sampling_bool) ? model.N_t : 1
    thermo_bool = num_temps > 1

    log_dist = prior_sampling_bool ? unadjusted_logprior : unadjusted_logpos
    xchange = (
        exchange_type != "none" && thermo_bool ?
            ReplicaXchange(Q, P, S, num_temps) : NoExchange()
    )
    kernel = UlaKernel(η, sqrt(2 * η), log_dist, xchange)

    return ULA_sampler(
        prior_sampling_bool, N,
        model, Q, P, S, num_temps, thermo_bool, kernel,
    )
end

function (sampler::ULA_sampler)(
        ps,
        st_kan,
        st_lux,
        x,
        st_rng;
        temps = [1.0f0],
    )
    """ULA posterior sampler. Returns z ~ p(z|x)."""
    model = sampler.model
    Q, P, S, num_temps = sampler.Q, sampler.P, sampler.S, sampler.num_temps
    prior_sampling_bool = sampler.prior_sampling_bool

    # Initialize from prior
    z_flat = begin
        if model.prior.bool_config.ula && prior_sampling_bool
            rv = st_rng.posterior_its
            rv = model.prior.prior_type == "lognormal" ? exp.(rv) : rv
            rv
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

    N_steps = sampler.N
    x_t = !prior_sampling_bool ? (
            model.lkhood.SEQ ? repeat(x, 1, 1, num_temps) :
            (model.use_pca ? repeat(x, 1, num_temps) : repeat(x, 1, 1, 1, num_temps))
        ) : nothing

    noise = st_rng.mcmc_noise
    log_u_swap = st_rng.log_swap
    mask_swap_1 = num_temps > 1 ? st_rng.swap_mask_1 : nothing
    mask_swap_2 = num_temps > 1 ? st_rng.swap_mask_2 : nothing

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
            log_u_swap,
            mask_swap_1,
            mask_swap_2,
            component_mask,
            temps,
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
