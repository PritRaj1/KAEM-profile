module MixtureChoice

export choose_component

using NNlib: softmax
using LinearAlgebra, Lux, Reactant

using ..Utils

function mask_kernel(α, rand_vals, Q, P, S)
    """One-hot mask at j = first p s.t. α[q,p] ≥ rv."""
    j = Reactant.Ops.findfirst(α .>= rand_vals; dimension = 2)
    return reshape(j, Q, 1, S) .== reshape(1:P, 1, P, 1) |> Lux.f32
end

function choose_component(α, num_samples, q_size, p_size, st_rng; ula_init = false)
    """One-hot (Q, P, num_samples) mask selecting one mixture component per (q, sample)."""
    rand_vals = ula_init ? st_rng.mix_rv_mcmc : st_rng.mix_rv
    α = cumsum(softmax(α; dims = 2); dims = 2)
    return mask_kernel(α, rand_vals, q_size, p_size, num_samples)
end

end
