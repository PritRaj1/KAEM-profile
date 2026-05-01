module InverseTransformSampling

export UnivITSSampler, MixITSSampler

using LinearAlgebra, ComponentArrays, Lux

using ..Utils

include("mixture_selection.jl")
using .MixtureChoice: choose_component

_init_rv(::MixtureMode, rand_vals) = rand_vals
_init_rv(::UnivariateMode, rand_vals) =
    PermutedDimsArray(view(rand_vals, :, :, :, :), (1, 2, 4, 3))

_final_rv(::MixtureMode, rand_vals) = dropdims(rand_vals; dims = 3)
_final_rv(::UnivariateMode, rand_vals) = rand_vals

function interpolate_kernel(
        cdf, grid, rand_vals, G;
        mode::AbstractSamplingMode = UnivariateMode(),
    )
    grid_idxs = reshape(1:(G + 1), 1, 1, G + 1, 1)
    grid = cat(grid, view(grid, :, :, G:G); dims = 3) # Repeat end, so G + 1 indexes final

    # First index, i, such that cdf[i] >= rand_vals
    rv = _init_rv(mode, rand_vals)
    indices = sum(cdf .< rv; dims = 3) .+ 1
    first_bool = indices .== 1 |> Lux.f32
    mask2 = indices .== grid_idxs |> Lux.f32
    mask1 = mask2 .- 1.0f0

    z1 = dropdims(
        (first_bool .* grid[:, :, 1]) .+
            (1.0f0 .- first_bool) .*
            sum(mask1 .* grid; dims = 3); dims = 3
    )
    z2 = dropdims(sum(mask2 .* grid; dims = 3); dims = 3)

    c1 = dropdims(
        (first_bool .* 0.0f0) .+
            (1.0f0 .- first_bool) .*
            sum(mask1 .* cdf; dims = 3); dims = 3
    )
    c2 = dropdims(sum(mask2 .* cdf; dims = 3); dims = 3)
    rv = _final_rv(mode, rand_vals)

    cdf_length = c2 - c1
    return ifelse.(
        cdf_length .== 0,
        z1,
        z1 .+ (z2 .- z1) .* ((rv .- c1) ./ cdf_length)
    )
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

struct UnivITSSampler{E}
    ebm::E
end

function (s::UnivITSSampler)(ps, st_kan, st_lux, st_rng; ula_init = false)
    ebm = s.ebm
    cdf, grid, st_lyrnorm_new = ebm.quad(
        ebm, ps.ebm, st_kan.ebm, Lux.testmode(st_lux.ebm), st_kan.quad,
    )
    cdf = cumsum(cdf; dims = 3)
    cdf = cat(view(zero(cdf), :, :, 1:1), cdf; dims = 3)

    rv = ula_init ? st_rng.posterior_its : st_rng.prior_its
    rand_vals = rv .* cdf[:, :, end]
    z = interpolate_kernel(
        cdf,
        PermutedDimsArray(view(grid, :, :, :), (3, 1, 2)),
        rand_vals,
        ebm.N_quad,
    )
    return z, st_lyrnorm_new
end

struct MixITSSampler{E}
    ebm::E
end

function (s::MixITSSampler)(ps, st_kan, st_lux, st_rng; ula_init = false)
    ebm = s.ebm
    alpha = ps.ebm.dist.α .* 1.0f0
    if ebm.bool_config.use_attention_kernel
        scale = sqrt(Float32(ebm.s_size))
        alpha = dotprod_attn(
            ps.ebm.attention.Q,
            ps.ebm.attention.K,
            st_rng.attn_rand,
            scale,
            st_kan.ebm[:a].min,
            st_kan.ebm[:a].max,
            ebm.q_size,
            ebm.s_size,
        )
    end
    mask = choose_component(
        alpha,
        ebm.s_size,
        ebm.q_size,
        ebm.p_size,
        st_rng;
        ula_init = ula_init,
    )
    cdf, grid, st_lyrnorm_new = ebm.quad(
        ebm, ps.ebm, st_kan.ebm, Lux.testmode(st_lux.ebm), st_kan.quad;
        mode = MixtureMode(),
        component_mask = mask,
    )

    cdf = cumsum(cdf; dims = 3)
    cdf = PermutedDimsArray(view(cdf, :, :, :, :), (1, 4, 3, 2))
    cdf = cat(view(zero(cdf), :, :, 1:1, :), cdf; dims = 3)

    rv = ula_init ? st_rng.posterior_its : st_rng.prior_its
    rand_vals = rv .* cdf[:, :, end:end, :]
    z = interpolate_kernel(
        cdf,
        PermutedDimsArray(view(grid, :, :, :), (1, 3, 2)),
        rand_vals,
        ebm.N_quad;
        mode = MixtureMode(),
    )
    return z, st_lyrnorm_new
end

end
