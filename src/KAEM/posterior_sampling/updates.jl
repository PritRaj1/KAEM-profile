module LangevinUpdates

export unadjusted_logpos, unadjusted_logprior, unadjusted_grad, per_sample_logpos

using ComponentArrays, Statistics, Lux, LinearAlgebra, Random, Enzyme

using ..Utils
using ..KAEM_model

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_MALA

# Per-sample log-densities
function per_sample_logpos(
        z,
        x,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
        component_mask,
        zero_vector,
    )

    logprior = first(
        model.log_prior(
            z,
            model.prior,
            ps.ebm,
            st_kan.ebm,
            st_lux.ebm,
            st_kan.quad;
            ula = true,
            component_mask = component_mask,
        )
    )

    ll = first(
        log_likelihood_MALA(
            z,
            x,
            model.lkhood,
            ps.gen,
            st_kan.gen,
            st_lux.gen,
            zero_vector;
            ε = model.ε,
        )
    )

    return logprior .+ temps .* ll
end

function per_sample_logprior(
        z,
        x,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
        component_mask,
        zero_vector,
    )

    return first(
        model.log_prior(
            z,
            model.prior,
            ps.ebm,
            st_kan.ebm,
            st_lux.ebm,
            st_kan.quad;
            ula = true,
            component_mask = component_mask,
        )
    )
end

# For autodiff
function unadjusted_logpos(
        z,
        x,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
        component_mask,
        zero_vector,
    )
    return sum(
        per_sample_logpos(
            z,
            x,
            temps,
            model,
            ps,
            st_kan,
            st_lux,
            component_mask,
            zero_vector,
        )
    )
end

function unadjusted_logprior(
        z,
        x,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
        component_mask,
        zero_vector,
    )
    return sum(
        per_sample_logprior(
            z,
            x,
            temps,
            model,
            ps,
            st_kan,
            st_lux,
            component_mask,
            zero_vector,
        )
    )
end

function unadjusted_grad(
        z,
        x,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
        component_mask,
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
            Enzyme.Const(component_mask),
            Enzyme.Const(zero_vector)
        )
    )
end

end
