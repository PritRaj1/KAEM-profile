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

include("updates.jl")
using .LangevinUpdates

include("sampler_init.jl")
using .SamplerInit

struct ULA_sampler
    prior_sampling_bool
    N
    η
    sqrt_2η
    model
    Q
    P
    S
    num_temps
    thermo_bool
    log_dist
end

function initialize_ULA_sampler(
        model::KAEM{T};
        η::T = 1.0f-3,
        prior_sampling_bool::Bool = false,
        N::Int = 20,
    ) where {T}

    Q, S = model.prior.q_size, model.batch_size
    P = model.prior.bool_config.mixture_model ? 1 : model.prior.p_size
    num_temps = (model.N_t > 1 && !prior_sampling_bool) ? model.N_t : 1
    thermo_bool = num_temps > 1
    log_dist = prior_sampling_bool ? unadjusted_logprior : unadjusted_logpos

    return ULA_sampler(
        prior_sampling_bool,
        N,
        η,
        sqrt(2 * η),
        model,
        Q,
        P,
        S,
        num_temps,
        thermo_bool,
        log_dist,
    )
end

function step(
        i,
        z_i,
        x_t,
        temps,
        temps_gpu,
        η,
        sqrt_2η,
        sampler,
        model,
        lkhood_copy,
        ps,
        st_kan,
        st_lux,
        noise,
        log_u_swap,
        mask_swap_1,
        mask_swap_2,
        component_mask,
        shift_down,
        shift_up,
    )
    ξ = noise[:, :, :, i]
    ∇z =
        unadjusted_grad(
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
    new_z = z_i .+ η .* ∇z .+ sqrt_2η .* ξ
    return model.xchange_func(
        i,
        new_z,
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

function (sampler::ULA_sampler)(
        ps,
        st_kan,
        st_lux,
        x,
        st_rng;
        temps = [1.0f0],
    )
    """ULA posterior sampler. Returns z ~ p(z|x)."""
    Q, P, S, num_temps = sampler.Q, sampler.P, sampler.S, sampler.num_temps

    ss = init_sampler_state(
        sampler.model, ps, st_kan, st_lux, st_rng, x,
        Q, P, S, num_temps;
        prior_sampling_bool = sampler.prior_sampling_bool,
    )
    model = ss.model
    temps_gpu = repeat(temps, S)

    state = (1, ss.z_flat)
    @trace while first(state) <= sampler.N
        i, z_acc = state
        z_new = step(
            i,
            z_acc,
            ss.x_t,
            temps,
            temps_gpu,
            sampler.η,
            sampler.sqrt_2η,
            sampler,
            model,
            ss.lkhood_copy,
            ps,
            st_kan,
            st_lux,
            ss.noise,
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

    if sampler.prior_sampling_bool
        st_lux = st_lux.ebm
        z = dropdims(z; dims = 4)
    end

    return z, st_lux
end


end
