module MixtureChoice

export choose_component

using NNlib: softmax
using Flux: onehotbatch
using LinearAlgebra, Random

using ..Utils

function mask_kernel!(
        mask,
        α,
        rand_vals,
        p_size,
    )
    for q in 1:size(rand_vals, 1), b in 1:size(rand_vals, 2)
        idx = p_size
        val = rand_vals[q, b]

        # Potential thread divergence on GPU
        for j in 1:p_size
            if α[q, j] >= val
                idx = j
                break
            end
        end

        # One-hot vector for this (q, b)
        for k in 1:p_size
            mask[q, k, b] = (idx == k) ? 1.0f0 : 0.0f0
        end
    end
    return nothing
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
    rand_vals = rand(Float32, rng, q_size, num_samples) |> pu
    α = cumsum(softmax(α; dims = 2); dims = 2)

    mask = zeros(Float32, q_size, p_size, num_samples) |> pu
    mask_kernel!(mask, α, rand_vals, p_size)
    return mask
end

end
