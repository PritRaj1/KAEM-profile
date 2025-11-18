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
        x,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
        prior_sampling_bool::Bool,
    )

    zero_vector = zeros(Float32, model.lkhood.x_shape..., size(z)[end])

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

function update_z!(
        z,
        ∇z,
        η,
        ξ,
        sqrt_2η,
    )
    @. z = z + η * ∇z + sqrt_2η * ξ
    return nothing
end

end
