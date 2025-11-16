module InverseTransformSampling

export sample_univariate, sample_mixture

using LinearAlgebra, Random, ComponentArrays, Lux

using ..Utils

include("mixture_selection.jl")
using .MixtureChoice: choose_component


function interpolate_kernel(cdf, grid, rand_vals)
    Q, P, G, S = size(cdf)..., size(rand_vals)[end]
    grid_idxs = reshape(1:G, 1, 1, G, 1) |> pu

    # First index, i, such that cdf[i] >= rand_vals
    indices = sum(1 .+ (cdf .< reshape(rand_vals, Q, P, 1, S)); dims = 3)
    mask2 = indices .== grid_idxs |> Lux.f32
    mask1 = mask2 .- 1.0f0
    first_bool = indices .== 1 |> Lux.f32

    z1 = dropdims((first_bool .* grid[:, :, 1]) .+ sum((1.0f0 .- first_bool) .* mask1 .* grid; dims = 3); dims = 3)
    z2 = dropdims(sum(mask2 .* grid; dims = 3); dims = 3)

    c1 = dropdims(first_bool .* 0.0f0 .+ sum((1.0f0 .- first_bool) .* mask1 .* cdf; dims = 3); dims = 3)
    c2 = dropdims(sum(mask2 .* cdf; dims = 3); dims = 3)
    length = c2 - c1
    return ifelse.(
        length .== 0,
        z1,
        z1 .+ (z2 .- z1) .* ((rand_vals .- c1) ./ (c2 .- c1))
    )
end

function sample_univariate(
        ebm,
        num_samples,
        ps,
        st_kan,
        st_lyrnorm,
        st_quad;
        rng = Random.default_rng(),
    )

    cdf, grid, st_lyrnorm_new = ebm.quad(ebm, ps, st_kan, st_lyrnorm, st_quad)

    cdf = cat(
        cumsum(cdf; dims = 3), # Cumulative trapezium = CDF
        dims = 3,
    )

    rand_vals = rand(rng, Float32, 1, ebm.p_size, num_samples) .* cdf[:, :, end]
    z = interpolate_kernel(
        cdf,
        reshape(grid, 1, size(grid)...),
        rand_vals,
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
        st_lyrnorm,
        st_quad;
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
    alpha = @view(ps.dist.α[:, :])
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
        ebm.quad(ebm, ps, st_kan, st_lyrnorm, st_quad; component_mask = mask, mix_bool = true)

    cdf = cat(
        cumsum(cdf; dims = 3), # Cumulative trapezium = CDF
        dims = 3,
    )

    rand_vals = rand(rng, Float32, ebm.q_size, num_samples) .* cdf[:, :, end]
    z = interpolate_kernel(
        cdf,
        reshape(grid, size(grid, 1), 1, size(grid, 2)),
        rand_vals,
    )
    return z, st_lyrnorm_new
end

end
