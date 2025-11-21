module LangevinUpdates

export unadjusted_logpos_grad, update_z!

using ComponentArrays, Statistics, Lux, LinearAlgebra, Random, Enzyme

using ..Utils
using ..KAEM_model

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_MALA

### ULA ###
function unadjusted_logpos(
        z,
        x,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
        num_temps,
        prior_sampling_bool::Bool,
        zero_vector,
    )

    log_posterior = 0.0f0
    for t in 1:num_temps
        lp = sum(
            first(
                model.log_prior(
                    view(z, :, :, :, t),
                    model.prior,
                    ps.ebm,
                    st_kan.ebm,
                    st_lux.ebm;
                    ula = true
                )
            )
        )

        ll = temps[t] * sum(
            first(
                log_likelihood_MALA(
                    view(z, :, :, :, t),
                    x,
                    model.lkhood,
                    ps.gen,
                    st_kan.gen,
                    st_lux.gen,
                    zero_vector;
                    ε = model.ε,
                )
            )
        )

        log_posterior += lp + ll
    end

    return log_posterior
end

function unadjusted_logpos_grad(
        z,
        x,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
        num_temps,
        prior_sampling_bool::Bool,
    )

    zero_vector = zeros(Float32, model.lkhood.x_shape..., size(x)[end])

    return first(
        Enzyme.gradient(
            Enzyme.Reverse,
            Enzyme.Const(unadjusted_logpos),
            z,
            Enzyme.Const(x),
            Enzyme.Const(temps),
            Enzyme.Const(model),
            Enzyme.Const(ps),
            Enzyme.Const(st_kan),
            Enzyme.Const(st_lux),
            Enzyme.Const(num_temps),
            Enzyme.Const(prior_sampling_bool),
            Enzyme.Const(zero_vector)
        )
    )
end

end
