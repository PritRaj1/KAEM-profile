module LangevinUpdates

export unadjusted_logpos, unadjusted_logprior, unadjusted_grad

using ComponentArrays, Statistics, Lux, LinearAlgebra, Random, Enzyme
using Reactant: @trace

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
        zero_vector,
    )

    log_posterior = 0.0f0
    @trace for t in 1:num_temps
        lp = sum(
            first(
                model.log_prior(
                    z[:, :, :, t],
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
                    z[:, :, :, t],
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

function unadjusted_logprior(
        z,
        x,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
        num_temps,
        zero_vector,
    )

    lp = 0.0f0
    @trace for t in 1:num_temps
        lp += sum(
            first(
                model.log_prior(
                    z[:, :, :, t],
                    model.prior,
                    ps.ebm,
                    st_kan.ebm,
                    st_lux.ebm;
                    ula = true
                )
            )
        )
    end

    return lp
end

function unadjusted_grad(
        z,
        x,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
        num_temps,
        log_dist,
    )

    zero_vector = zeros(Float32, model.lkhood.x_shape..., size(x)[end])

    return first(
        Enzyme.gradient(
            Enzyme.Reverse,
            Enzyme.Const(log_dist),
            z,
            Enzyme.Const(x),
            Enzyme.Const(temps),
            Enzyme.Const(model),
            Enzyme.Const(ps),
            Enzyme.Const(st_kan),
            Enzyme.Const(st_lux),
            Enzyme.Const(num_temps),
            Enzyme.Const(zero_vector)
        )
    )
end

end
