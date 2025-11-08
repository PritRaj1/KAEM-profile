module LogPosteriors

using ComponentArrays, Statistics, Lux, LinearAlgebra, Random, Enzyme

using ..Utils
using ..T_KAM_model

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
        prior_sampling_bool::Bool,
        zero_vector,
    )
    lp = sum(
        first(model.log_prior(z, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm; ula = true)),
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
        ),
    )
    tempered_ll = sum(temps .* ll)
    return (lp + tempered_ll)
end

function unadjusted_logpos_grad(
        z,
        ∇z,
        x,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
        prior_sampling_bool::Bool,
    )

    zero_vector = zeros(Float32, model.lkhood.x_shape..., size(z)[end]) |> pu

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
            Enzyme.Const(prior_sampling_bool),
            Enzyme.Const(zero_vector)
        )
    )
end

### autoMALA ###
function autoMALA_logpos(
        z,
        x,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
        zero_vector,
    )
    st_ebm, st_gen = st_kan.ebm, st_lux.gen
    lp, st_ebm = model.log_prior(z, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm; ula = true)
    ll, st_gen = log_likelihood_MALA(
        z,
        x,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux.gen,
        zero_vector;
        ε = model.ε,
    )
    return (lp + temps .* ll), st_ebm, st_gen
end

function closure(
        z,
        x,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
        zero_vector,
    )
    return sum(first(autoMALA_logpos(z, x, temps, model, ps, st_kan, st_lux, zero_vector)))
end

function autoMALA_value_and_grad(
        z,
        ∇z,
        x,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
    )
    zero_vector = zeros(Float32, model.lkhood.x_shape..., size(z)[end]) |> pu

    ∇z = first(
        Enzyme.gradient(
            Enzyme.Reverse,
            Enzyme.Const(closure),
            z,
            Enzyme.Const(x),
            Enzyme.Const(temps),
            Enzyme.Const(model),
            Enzyme.Const(ps),
            Enzyme.Const(st_kan),
            Enzyme.Const(st_lux),
            Enzyme.Const(zero_vector)
        )
    )

    logpos, st_ebm, st_gen =
        autoMALA_logpos(z, x, temps, model, ps, st_kan, st_lux, zero_vector)
    return logpos, ∇z, st_ebm, st_gen
end

end
