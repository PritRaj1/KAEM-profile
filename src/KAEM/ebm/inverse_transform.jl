module InverseTransformSampling

export UnivITSSampler, MixITSSampler

using LinearAlgebra, ComponentArrays, Lux, Reactant

using ..Utils

include("mixture_selection.jl")
using .MixtureChoice: choose_component

function dotprod_attn(Q, K, z, scale, min_z, max_z, q_size, s_size)
    z = reshape(z, q_size, 1, s_size) .* ((max_z - min_z) + min_z)
    return dropdims(sum((Q .* z) .* (K .* z); dims = 3); dims = 3) ./ scale
end

# Linear interp z(rv) on bin (z1,c1)->(z2,c2)
function _lerp(z1, z2, c1, c2, rv)
    len = c2 .- c1
    safe = ifelse.(len .== 0, 1.0f0, len)
    return z1 .+ (z2 .- z1) .* ((rv .- c1) ./ safe)
end

function _its_step(cdf, grid, rv, Q, M, B, mixture::Bool)
    jh = Reactant.Ops.findfirst(cdf .>= rv; dimension = 3)
    jl = max.(jh .- Int64(1), Int64(1))

    Mtot = Q * M * B
    qcol = reshape(Reactant.Ops.iota(Int64, [Q, M, B]; iota_dimension = 1), Mtot, 1)
    mcol = reshape(Reactant.Ops.iota(Int64, [Q, M, B]; iota_dimension = 2), Mtot, 1)
    bcol = reshape(Reactant.Ops.iota(Int64, [Q, M, B]; iota_dimension = 3), Mtot, 1)
    one = Reactant.Ops.fill(Int64(1), [Mtot, 1])
    jhc, jlc = reshape(jh, Mtot, 1), reshape(jl, Mtot, 1)

    cdf_b = mixture ? bcol : one
    grid_q = mixture ? qcol : one

    out = (Q, M, B)
    gather(src, idx) = reshape(Reactant.Ops.gather_getindex(src, idx), out)

    z2 = gather(grid, hcat(grid_q, mcol, jhc, one))
    z1 = gather(grid, hcat(grid_q, mcol, jlc, one))
    c2 = gather(cdf, hcat(qcol, mcol, jhc, cdf_b))
    c1 = gather(cdf, hcat(qcol, mcol, jlc, cdf_b))

    return _lerp(z1, z2, c1, c2, dropdims(rv; dims = 3))
end

struct UnivITSSampler{E}
    ebm::E
end

function (s::UnivITSSampler)(ps, st_kan, st_lux, st_rng; ula_init = false)
    ebm = s.ebm
    Q, P, B, G = ebm.q_size, ebm.p_size, ebm.s_size, ebm.N_quad

    cdf, grid, st_lyrnorm_new = ebm.quad(
        ebm, ps.ebm, st_kan.ebm, Lux.testmode(st_lux.ebm), st_kan.quad,
    )
    cdf = cumsum(cdf; dims = 3)
    cdf = cat(view(zero(cdf), :, :, 1:1), cdf; dims = 3)

    zmin = ebm.bool_config.no_grid ? st_kan.ebm[:a].min : view(st_kan.ebm[:a].grid, :, 1)
    grid = PermutedDimsArray(view(grid, :, :, :), (3, 1, 2))
    grid = cat(reshape(zmin, 1, P, 1), grid; dims = 3)

    rv = ula_init ? st_rng.posterior_its : st_rng.prior_its
    rand_vals = rv .* cdf[:, :, end]

    cdf_4d = reshape(cdf, Q, P, G + 1, 1) .* 1.0f0
    grid_4d = reshape(grid, 1, P, G + 1, 1) .* 1.0f0
    rv_4d = reshape(rand_vals, Q, P, 1, B) .* 1.0f0

    return _its_step(cdf_4d, grid_4d, rv_4d, Q, P, B, false), st_lyrnorm_new
end

struct MixITSSampler{E}
    ebm::E
end

function (s::MixITSSampler)(ps, st_kan, st_lux, st_rng; ula_init = false)
    ebm = s.ebm
    Q, P, B, G = ebm.q_size, ebm.p_size, ebm.s_size, ebm.N_quad

    alpha = ebm.bool_config.use_attention_kernel ?
        dotprod_attn(
            ps.ebm.attention.Q, ps.ebm.attention.K, st_rng.attn_rand,
            sqrt(Float32(B)), st_kan.ebm[:a].min, st_kan.ebm[:a].max, Q, B,
        ) : ps.ebm.dist.α .* 1.0f0
    mask = choose_component(alpha, B, Q, P, st_rng; ula_init = ula_init)

    cdf, grid, st_lyrnorm_new = ebm.quad(
        ebm, ps.ebm, st_kan.ebm, Lux.testmode(st_lux.ebm), st_kan.quad;
        mode = MixtureMode(), component_mask = mask,
    )

    cdf = cumsum(cdf; dims = 3)
    cdf = PermutedDimsArray(view(cdf, :, :, :, :), (1, 4, 3, 2))
    cdf = cat(view(zero(cdf), :, :, 1:1, :), cdf; dims = 3)

    zmin = ebm.bool_config.no_grid ? st_kan.ebm[:a].min : view(st_kan.ebm[:a].grid, :, 1)
    grid = PermutedDimsArray(view(grid, :, :, :), (1, 3, 2))
    grid = cat(reshape(zmin, Q, 1, 1), grid; dims = 3)

    rv = ula_init ? st_rng.posterior_its : st_rng.prior_its
    rand_vals = rv .* cdf[:, :, end:end, :] .* 1.0f0
    cdf_4d = cdf .* 1.0f0
    grid_4d = reshape(grid, Q, 1, G + 1, 1) .* 1.0f0

    return _its_step(cdf_4d, grid_4d, rand_vals, Q, 1, B, true), st_lyrnorm_new
end

end
