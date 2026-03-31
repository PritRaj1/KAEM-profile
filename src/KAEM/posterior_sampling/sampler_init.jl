module SamplerInit

export init_sampler_state

using Accessors

using ..Utils
using ..KAEM_model
using ..KAEM_model.InverseTransformSampling

include("../ebm/mixture_selection.jl")
using .MixtureChoice: choose_component

function init_sampler_state(
        model, ps, st_kan, st_lux, st_rng, x,
        Q, P, S, num_temps;
        prior_sampling_bool = false,
    )

    # Initialize z from prior
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

    lkhood_copy = model.lkhood

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

    x_t = !prior_sampling_bool ? (
            model.lkhood.SEQ ? repeat(x, 1, 1, num_temps) :
            (model.use_pca ? repeat(x, 1, num_temps) : repeat(x, 1, 1, 1, num_temps))
        ) : nothing

    noise = st_rng.ula_noise
    log_u_swap = st_rng.log_swap
    mask_swap_1 = num_temps > 1 ? st_rng.swap_mask_1 : nothing
    mask_swap_2 = num_temps > 1 ? st_rng.swap_mask_2 : nothing
    shift_down = num_temps > 1 ? st_rng.shift_down : nothing
    shift_up = num_temps > 1 ? st_rng.shift_up : nothing

    return (
        model = model,
        z_flat = z_flat,
        lkhood_copy = lkhood_copy,
        component_mask = component_mask,
        x_t = x_t,
        noise = noise,
        log_u_swap = log_u_swap,
        mask_swap_1 = mask_swap_1,
        mask_swap_2 = mask_swap_2,
        shift_down = shift_down,
        shift_up = shift_up,
    )
end

end
