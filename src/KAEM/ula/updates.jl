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

    logpos = 0.0f0
    @trace for t in 1:num_temps
        logpos += sum(
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

        temp = sum(temps[t:t, :])
        logpos += temp * sum(
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
    end

    return logpos
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

    zero_vector = zero(x)

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
