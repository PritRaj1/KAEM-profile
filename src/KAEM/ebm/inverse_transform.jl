module InverseTransformSampling

export sample_univariate, sample_mixture

using LinearAlgebra, Random, ComponentArrays

using ..Utils

include("mixture_selection.jl")
using .MixtureChoice: choose_component

function interpolate_single(
        idx,
        cdf,
        grid,
        rv,
    )
    idx <= 1 && return grid[1]
    idx > size(grid, 2) && return grid[end]
    z1, z2 = grid[idx - 1], grid[idx]
    cd1, cd2 = cdf[idx - 1], cdf[idx]
    length = cd2 - cd1
    length == 0 && return z1
    return z1 + (z2 - z1) * ((rv - cd1) / length)
end

function interpolate_batch(
        cdf,
        grid,
        rv,
    )
    indices = searchsortedfirst.(Ref(cdf), rv)
    return interpolate_single.(indices, Ref(cdf), Ref(grid), rv)
end

function interpolate_p(
        cdf,
        grid,
        rv,
        P
    )
    z = reduce(
        hcat,
        map(
            p -> interpolate_batch(
                selectdim(cdf, 1, p),
                selectdim(grid, 1, p),
                selectdim(rv, 1, p)
            ), 1:P
        )
    )
    return reshape(z, 1, P, size(rv, 2))
end

function interpolate_kernel(
        cdf,
        grid,
        rand_vals,
        Q,
        P
    )
    return reduce(
        vcat,
        map(
            q -> interpolate_p(
                selectdim(cdf, 1, q),
                grid,
                selectdim(rand_vals, 1, q), P
            ), 1:Q
        )
    )
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
    grid_size = size(grid, 2)

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
