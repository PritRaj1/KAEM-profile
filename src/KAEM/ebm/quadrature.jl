module Quadrature

export TrapeziumQuadrature, GaussLegendreQuadrature

using LinearAlgebra, Random, ComponentArrays

using ..Utils

negative_one = - ones(Float32, 1, 1, 1) |> pu

struct TrapeziumQuadrature <: AbstractQuadrature end

struct GaussLegendreQuadrature <: AbstractQuadrature end

function qfirst_exp_kernel(f, π0)
    return exp.(f) .* reshape(π0, size(π0, 1), 1, size(π0, 2))
end

function pfirst_exp_kernel(f, π0)
    return exp.(f) .* reshape(π0, 1, size(π0)...)
end

function apply_mask(exp_fg, component_mask)
    return dropdims(sum(permutedims(exp_fg[:, :, :, :], (1, 2, 4, 3)) .* component_mask; dims = 2); dims = 2)
end

function weight_kernel(trapz, weights)
    return reshape(weights, 1, size(weights)...) .* trapz
end

function gauss_kernel(
        trapz,
        weights,
    )
    return reshape(weights, size(weights, 1), 1, size(weights, 2)) .* trapz
end

function (tq::TrapeziumQuadrature)(
        ebm,
        ps,
        st_kan,
        st_lyrnorm::NamedTuple;
        component_mask = negative_one,
    )
    """Trapezoidal rule for numerical integration: 1/2 * (u(z_{i-1}) + u(z_i)) * Δx"""

    # Evaluate prior on grid [0,1]
    f_grid = st_kan[:a].grid
    Δg = f_grid[:, 2:end] - f_grid[:, 1:(end - 1)]

    I, O = size(f_grid)
    π_grid = ebm.π_pdf(f_grid[:, :, :], ps.dist.π_μ, ps.dist.π_σ)
    π_grid =
        ebm.prior_type == "learnable_gaussian" ? dropdims(π_grid, dims = 3)' :
        dropdims(π_grid, dims = 3)

    # Energy function of each component
    f_grid, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, f_grid)
    Q, P, G = size(f_grid)

    # Choose component if mixture model else use all
    if !any(component_mask .< 0.0f0)
        B = size(component_mask, 3)
        exp_fg = qfirst_exp_kernel(f_grid, π_grid)
        trapz = apply_mask(exp_fg, component_mask)
        trapz = trapz[:, :, 2:end] + trapz[:, :, 1:(end - 1)]
        trapz = weight_kernel(trapz, Δg)
        return trapz ./ 2, st_kan[:a].grid, st_lyrnorm_new
    else
        exp_fg = pfirst_exp_kernel(f_grid, π_grid)
        trapz = exp_fg[:, :, 2:end] + exp_fg[:, :, 1:(end - 1)]
        trapz = weight_kernel(trapz, Δg)
        return trapz ./ 2, st_kan[:a].grid, st_lyrnorm_new
    end
end

function get_gausslegendre(
        ebm,
        ps,
        st_kan,
    )
    """Get Gauss-Legendre nodes and weights for prior's domain"""

    a, b = minimum(st_kan[:a].grid; dims = 2), maximum(st_kan[:a].grid; dims = 2)

    no_grid =
        (ebm.fcns_qp[1].spline_string == "FFT" || ebm.fcns_qp[1].spline_string == "Cheby")

    if no_grid
        a = fill(Float32(first(ebm.prior_domain)), size(a)) |> pu
        b = fill(Float32(last(ebm.prior_domain)), size(b)) |> pu
    end

    return ((a + b) / 2 + (b - a) / 2) .* ebm.nodes, (b - a) ./ 2 .* ebm.weights
end

function (gq::GaussLegendreQuadrature)(
        ebm,
        ps,
        st_kan,
        st_lyrnorm;
        component_mask = negative_one,
    )
    """Gauss-Legendre quadrature for numerical integration"""

    nodes, weights = get_gausslegendre(ebm, ps, st_kan)
    grid = nodes

    I, O = size(nodes)
    π_nodes = ebm.π_pdf(nodes[:, :, :], ps.dist.π_μ, ps.dist.π_σ)
    π_nodes =
        ebm.prior_type == "learnable_gaussian" ? dropdims(π_nodes, dims = 3)' :
        dropdims(π_nodes, dims = 3)

    # Energy function of each component
    nodes, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, nodes)
    Q, P, G = size(nodes)

    # Choose component if mixture model else use all
    if !any(component_mask .< 0.0f0)
        B = size(component_mask, 3)
        exp_fg = qfirst_exp_kernel(nodes, π_nodes)
        trapz = apply_mask(exp_fg, component_mask)
        trapz = gauss_kernel(trapz, weights)
        return trapz, grid, st_lyrnorm_new
    else
        exp_fg = pfirst_exp_kernel(nodes, π_nodes)
        exp_fg = weight_kernel(exp_fg, weights)
        return exp_fg, grid, st_lyrnorm_new
    end
end

end
