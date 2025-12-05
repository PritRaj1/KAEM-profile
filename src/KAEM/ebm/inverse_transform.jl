module InverseTransformSampling

export sample_univariate, sample_mixture

using LinearAlgebra, ComponentArrays, Lux

using ..Utils

include("mixture_selection.jl")
using .MixtureChoice: choose_component


function interpolate_kernel(cdf, grid, rand_vals, Q, P, G, S; mix_bool = false)
    grid_idxs = reshape(1:G, 1, 1, G, 1)

    # First index, i, such that cdf[i] >= rand_vals
    indices = sum(1 .+ (cdf .< reshape(rand_vals, Q, P, 1, S)); dims = 3)
    first_bool = indices .== 1 |> Lux.f32
    mask2 = indices .== grid_idxs |> Lux.f32
    mask1 = mask2 .- 1.0f0

    z1 = dropdims((first_bool .* grid[:, :, 1]) .+ (1.0f0 .- first_bool) .* sum(mask1 .* grid; dims = 3); dims = 3)
    z2 = dropdims(sum(mask2 .* grid; dims = 3); dims = 3)

    c1 = dropdims((first_bool .* 0.0f0) .+ (1.0f0 .- first_bool) .* sum(mask1 .* cdf; dims = 3); dims = 3)
    c2 = dropdims(sum(mask2 .* cdf; dims = 3); dims = 3)
    rv = mix_bool ? dropdims(rand_vals; dims = 3) : rand_vals
    length = c2 - c1

    return ifelse.(
        length .== 0,
        z1,
        z1 .+ (z2 .- z1) .* ((rv .- c1) ./ length)
    )
end

function sample_univariate(
        ebm,
        ps,
        st_kan,
        st_lyrnorm,
        st_quad,
        st_rng;
        ula_init = false
    )

    cdf, grid, st_lyrnorm_new = ebm.quad(ebm, ps, st_kan, st_lyrnorm, st_quad)
    cdf = cumsum(cdf; dims = 3) # Cumulative trapezium = CDF

    rv = ula_init ? st_rng.posterior_its : st_rng.prior_its
    rand_vals = rv .* cdf[:, :, end]
    z = interpolate_kernel(
        cdf,
        PermutedDimsArray(view(grid, :, :, :), (3, 1, 2)),
        rand_vals,
        ebm.q_size,
        ebm.p_size,
        ebm.N_quad,
        ebm.s_size
    )
    return z, st_lyrnorm_new
end

function dotprod_attn(
        Q,
        K,
        z,
        scale,
        min_z,
        max_z,
        q_size,
        s_size
    )
    z = reshape(z, q_size, 1, s_size) .* ((max_z - min_z) + min_z)
    return dropdims(sum((Q .* z) .* (K .* z); dims = 3); dims = 3) ./ scale
end

function sample_mixture(
        ebm,
        ps,
        st_kan,
        st_lyrnorm,
        st_quad,
        st_rng;
        ula_init = false
    )
    """
    Component-wise inverse transform sampling for the ebm-prior.
    p = components of model
    q = number of models

    Args:
        prior: The ebm-prior.
        ps: The parameters of the ebm-prior.
        st: The states of the ebm-prior.

    Returns:
        z: The samples from the ebm-prior, (num_samples, q). 
    """
    alpha = ps.dist.α .* 1.0f0
    if ebm.bool_config.use_attention_kernel
        scale = sqrt(Float32(ebm.s_size))
        alpha = dotprod_attn(
            ps.attention.Q,
            ps.attention.K,
            st_rng.attn_rand,
            scale,
            st_kan[:a].min,
            st_kan[:a].max,
            ebm.q_size,
            ebm.s_size
        )
    end
    mask = choose_component(alpha, ebm.s_size, ebm.q_size, ebm.p_size, st_rng; ula_init = ula_init)
    cdf, grid, st_lyrnorm_new =
        ebm.quad(ebm, ps, st_kan, st_lyrnorm, st_quad; component_mask = mask, mix_bool = true)
    cdf = cumsum(cdf; dims = 3) # Cumulative trapezium = CDF
    cdf = PermutedDimsArray(view(cdf, :, :, :, :), (1, 4, 3, 2))

    rv = ula_init ? st_rng.posterior_its : st_rng.prior_its
    rand_vals = rv .* cdf[:, :, end:end, :]
    z = interpolate_kernel(
        cdf,
        PermutedDimsArray(view(grid, :, :, :), (1, 3, 2)),
        rand_vals,
        ebm.q_size,
        1, # Single component chosen already
        ebm.N_quad,
        ebm.s_size;
        mix_bool = true
    )
    return z, st_lyrnorm_new
end

end
