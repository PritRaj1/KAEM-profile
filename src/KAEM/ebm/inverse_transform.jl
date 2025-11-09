module InverseTransformSampling

export sample_univariate, sample_mixture

using LinearAlgebra, Random, ComponentArrays

using ..Utils

include("mixture_selection.jl")
using .MixtureChoice: choose_component

function interp_kernel!(
        z,
        cdf,
        grid,
        rand_vals,
        grid_size,
    )
    for q in 1:size(z, 1), p in 1:size(z, 2), b in 1:size(z, 3)
        rv = rand_vals[q, p, b]
        idx = 1

        # Manual searchsortedfirst over cdf[q, p, :] - potential thread divergence on GPU
        for j in 1:(grid_size + 1)
            if cdf[q, p, j] >= rv
                idx = j
                break
            end
            idx = j
        end

        # Edge case 1: Random value is smaller than first CDF value
        if idx == 1
            z[q, p, b] = grid[p, 1]

            # Edge case 2: Random value is larger than last CDF value
        elseif idx > grid_size
            z[q, p, b] = grid[p, grid_size]

            # Interpolate into interval
        else
            z1, z2 = grid[p, idx - 1], grid[p, idx]
            cd1, cd2 = cdf[q, p, idx - 1], cdf[q, p, idx]

            # Handle exact match without instability
            length = cd2 - cd1
            if length == 0
                z[q, p, b] = z1
            else
                z[q, p, b] = z1 + (z2 - z1) * ((rv - cd1) / length)
            end
        end
    end
    return nothing
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

    rand_vals = pu(rand(rng, Float32, 1, ebm.p_size, num_samples)) .* cdf[:, :, end]
    z = zeros(Float32, ebm.q_size, ebm.p_size, num_samples) |> pu
    interp_kernel!(
        z,
        cdf,
        grid,
        rand_vals,
        grid_size,
    )
    return z, st_lyrnorm_new
end

function interp_kernel_mixture!(
        z,
        cdf,
        grid,
        rand_vals,
        grid_size,
    )
    for q in 1:size(rand_vals, 1), b in 1:size(rand_vals, 2)
        rv = rand_vals[q, b]
        idx = 1

        # Manual searchsortedfirst over cdf[q, b, :] - potential thread divergence on GPU
        for j in 1:(grid_size + 1)
            if cdf[q, b, j] >= rv
                idx = j
                break
            end
            idx = j
        end

        # Edge case 1: Random value is smaller than first CDF value
        if idx == 1
            z[q, 1, b] = grid[q, 1]

            # Edge case 2: Random value is larger than last CDF value
        elseif idx > grid_size
            z[q, 1, b] = grid[q, grid_size]

            # Interpolate into interval
        else
            z1, z2 = grid[q, idx - 1], grid[q, idx]
            cd1, cd2 = cdf[q, b, idx - 1], cdf[q, b, idx]

            # Handle exact match without instability
            length = cd2 - cd1
            if length == 0
                z[q, 1, b] = z1
            else
                z[q, 1, b] = z1 + (z2 - z1) * ((rv - cd1) / length)
            end
        end
    end
    return nothing
end

function dotprod_attn_kernel!(
        QK,
        Q,
        K,
        z,
        scale,
        min_z,
        max_z,
    )
    for q in 1:size(z, 1), p in 1:size(K, 2), b in 1:size(z, 2)
        z_mapped = z[q, b] * (max_z - min_z) + min_z
        QK[q, p] = ((Q[q, p] * z_mapped) * (K[q, p] * z_mapped)) / scale
    end
    return nothing
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
        z = rand(Float32, rng, ebm.q_size, num_samples) |> pu
        alpha = zeros(Float32, ebm.q_size, ebm.p_size) |> pu
        scale = sqrt(Float32(num_samples))
        min_z, max_z = ebm.prior_domain
        dotprod_attn_kernel!(
            alpha,
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

    rand_vals = pu(rand(rng, Float32, ebm.q_size, num_samples)) .* cdf[:, :, end]

    z = zeros(Float32, ebm.q_size, 1, num_samples) |> pu
    interp_kernel_mixture!(
        z,
        cdf,
        grid,
        rand_vals,
        grid_size,
    )
    return z, st_lyrnorm_new
end

end
