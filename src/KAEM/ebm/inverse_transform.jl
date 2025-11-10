module InverseTransformSampling

export sample_univariate, sample_mixture

using LinearAlgebra, Random, ComponentArrays

using ..Utils

include("mixture_selection.jl")
using .MixtureChoice: choose_component

function interpolate_single(
        cdf,
        grid,
        rv,
    )
    idx = searchsortedfirst(cdf, rv)
    early_return = ifelse(idx <= 1, grid[1], grid[end])
    early_bool = (idx <= 1) + (idx > size(grid, 2))

    z1, z2 = view(grid, max(1, idx - 1)), view(grid, min(size(grid, 2), idx))
    cd1, cd2 = view(cdf, max(1, idx - 1)), view(cdf, min(size(cdf, 2), idx))
    length = cd2 - cd1
    inter_return = ifelse(length == 0, z1, z1 + (z2 - z1) * ((rv - cd1) / length))
    return ifelse(early_bool, early_return, inter_return)
end

function interpolate_batch(
        cdf,
        grid,
        rv,
    )
    return interpolate_single.(Ref(cdf), Ref(grid), rv)
end

function interpolate_kernel(
        cdf,
        grid,
        rand_vals,
        Q,
        P
    )
    cdf_flat = reshape(cdf, Q * P, size(cdf, 3))
    grid_exp = repeat(grid, Q, 1)
    rand_vals_flat = reshape(rand_vals, Q * P, size(rand_vals, 3))
    z_flat = reduce(
        vcat,
        map(
            n -> interpolate_batch(
                view(cdf_flat, n, :),
                view(grid_exp, n, :),
                view(rand_vals_flat, n, :)
            ), 1:(Q * P)
        )
    )
    return reshape(z_flat, Q, P, size(rand_vals, 3))
end

function sample_univariate(
        ebm,
        num_samples,
        ps,
        st_kan,
        st_lyrnorm;
        rng = Random.default_rng(),
    )

    cdf, grid, st_lyrnorm_new = ebm.quad(ebm, ps, st_kan, st_lyrnorm)

    cdf = cat(
        cdf[:, :, 1:1] .* 0, # Add 0 to start of CDF
        cumsum(cdf; dims = 3), # Cumulative trapezium = CDF
        dims = 3,
    )

    rand_vals = rand(rng, Float32, 1, ebm.p_size, num_samples) .* cdf[:, :, end]
    z = interpolate_kernel(
        cdf,
        grid,
        rand_vals,
        ebm.q_size,
        ebm.p_size,
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
    )
    z = reshape(z, size(z, 1), 1, size(z, 2)) .* ((max_z - min_z) + min_z)
    return dropdims(sum((Q .* z) .* (K .* z); dims = 3); dims = 3) ./ scale
end

function sample_mixture(
        ebm,
        num_samples,
        ps,
        st_kan,
        st_lyrnorm;
        rng = Random.default_rng(),
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
    alpha = ps.dist.α
    if ebm.bool_config.use_attention_kernel
        z = rand(rng, Float32, ebm.q_size, num_samples)
        alpha = similar(ebm.p_size, ebm.q_size, ebm.p_size) .* 0
        scale = sqrt(Float32(num_samples))
        min_z, max_z = ebm.prior_domain
        alpha = dotprod_attn(
            ps.attention.Q,
            ps.attention.K,
            z,
            scale,
            min_z,
            max_z,
        )
    end
    mask = choose_component(alpha, num_samples, ebm.q_size, ebm.p_size; rng = rng)
    cdf, grid, st_lyrnorm_new =
        ebm.quad(ebm, ps, st_kan, st_lyrnorm; component_mask = mask)
    grid_size = size(grid, 2)

    cdf = cat(
        cdf[:, :, 1:1] .* 0, # Add 0 to start of CDF
        cumsum(cdf; dims = 3), # Cumulative trapezium = CDF
        dims = 3,
    )

    rand_vals = rand(rng, Float32, ebm.q_size, num_samples) .* cdf[:, :, end]
    z = interpolate_kernel(
        cdf,
        grid,
        rand_vals,
        ebm.q_size,
        1,
    )
    return z, st_lyrnorm_new
end

end
