module LogLikelihoods

export log_likelihood_IS, log_likelihood_MALA

using ComponentArrays, Random
using NNlib: softmax, sigmoid

using ..Utils
using ..KAEM_model: GenModel

include("losses.jl")
using .Losses

## Log-likelihood functions ##
function log_likelihood_IS(
        z,
        x,
        lkhood,
        ps,
        st_kan,
        st_lux,
        noise;
        ε = eps(Float32),
    )
    """
    Conditional likelihood of the generator.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data.
        z: The latent variable.
        tempered: Whether to use tempered likelihood.
        rng: The random number generator.

    Returns:
        The unnormalized log-likelihood.
    """
    B, S = size(x)[end], size(z)[end]
    x̂, st_lux_new = lkhood.generator(ps, st_kan, st_lux, z)
    noise_scaled = lkhood.σ.noise .* noise
    x̂_noised = lkhood.output_activation(x̂ .+ noise_scaled)

    ll = IS_loss(x, x̂_noised, ε, 2 * lkhood.σ.llhood^2, B, S, lkhood.SEQ)
    return ll, st_lux_new
end

function log_likelihood_MALA(
        z,
        x,
        lkhood,
        ps,
        st_kan,
        st_lux,
        noise;
        ε = eps(Float32),
    )
    """
    Conditional likelihood of the generator sampled by Langevin.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data.
        z: The latent variable.
        rng: The random number generator.

    Returns:
        The unnormalized log-likelihood.
    """
    B = size(z)[end]
    x̂, st_lux_new = lkhood.generator(ps, st_kan, st_lux, z)
    noise_scaled = lkhood.σ.noise .* noise
    x̂_act = lkhood.output_activation(x̂ .+ noise_scaled)

    ll =
        MALA_loss(x, x̂_act, ε, 2 * lkhood.σ.llhood^2, B, lkhood.SEQ, lkhood.perceptual_scale)
    return ll, st_lux_new
end

end
