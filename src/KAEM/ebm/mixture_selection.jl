module MixtureChoice

export choose_component

using NNlib: softmax
using LinearAlgebra, Random

using ..Utils

function mask_kernel(
        α,
        rand_vals,
    )
    """One-hot mask for chosen index"""
    Q, P, S = size(α)..., size(rand_vals, 2)
    indices = sum(1 .+ (α .< reshape(rand_vals, Q, 1, S)); dims = 2)
    p_range = reshape(1:P, 1, P, 1) |> pu
    mask = indices .== p_range |> Lux.f32
    return mask
end

function choose_component(
        α,
        num_samples,
        q_size,
        p_size;
        rng = Random.default_rng(),
    )
    """
    Creates a one-hot mask for mixture model, q, to select one component, p.

    Args:
        alpha: The mixture proportions, (q, p).
        num_samples: The number of samples to generate.
        q_size: The number of mixture models.
        rng: The random number generator.

    Returns:
        chosen_components: The one-hot mask for each mixture model, (num_samples, q, p).    
    """
    rand_vals = rand(rng, Float32, q_size, num_samples)
    α = cumsum(softmax(α; dims = 2); dims = 2)
    return mask_kernel(α, rand_vals)
end

end
