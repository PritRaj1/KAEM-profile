module Quadrature

export GaussLegendreQuadrature, get_gausslegendre

using LinearAlgebra, Random, ComponentArrays, FastGaussQuadrature

using ..Utils

negative_one = - ones(Float32, 1, 1, 1)

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

function get_gausslegendre(
        ebm,
        st_kan,
    )
    """Get Gauss-Legendre nodes and weights for prior's domain"""
    a, b = st_kan[:a].grid[:, 1], st_kan[:a].grid[:, end]
    no_grid =
        (ebm.fcns_qp[1].spline_string == "FFT" || ebm.fcns_qp[1].spline_string == "Cheby")

    if no_grid
        a .= a .* 0.0f0 .+ first(ebm.prior_domain)
        b .= b .* 0.0f0 .+ last(ebm.prior_domain)
    end

    nodes, weights = gausslegendre(ebm.N_quad)
    nodes = ((a .+ b) ./ 2 .+ (b .- a) ./ 2) * nodes'
    weights = ((b .- a) ./ 2) * weights'
    return nodes, weights
end

function mix_return(nodes, π_nodes, weights, component_mask)
    B = size(component_mask, 3)
    exp_fg = qfirst_exp_kernel(nodes, π_nodes)
    trapz = apply_mask(exp_fg, component_mask)
    trapz = gauss_kernel(trapz, weights)
    return trapz
end

function univar_return(nodes, π_nodes, weights)
    exp_fg = pfirst_exp_kernel(nodes, π_nodes)
    exp_fg = weight_kernel(exp_fg, weights)
    return exp_fg
end

function (gq::GaussLegendreQuadrature)(
        ebm,
        ps,
        st_kan,
        st_lyrnorm,
        st_quad;
        component_mask = negative_one,
        mix_bool::Bool = false,
    )
    """Gauss-Legendre quadrature for numerical integration"""

    nodes, weights = st_quad.nodes, st_quad.weights
    grid = nodes

    I, O = size(nodes)
    π_nodes = ebm.π_pdf(reshape(nodes, size(nodes)..., 1), ps.dist.π_μ, ps.dist.π_σ)
    π_nodes =
        ebm.prior_type == "learnable_gaussian" ? dropdims(π_nodes, dims = 3)' :
        dropdims(π_nodes, dims = 3)

    # Energy function of each component
    nodes, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, nodes)
    Q, P, G = size(nodes)

    # Choose component if mixture model else use all
    result = (
        mix_bool ?
            mix_return(nodes, π_nodes, weights, component_mask) :
            univar_return(nodes, π_nodes, weights)
    )

    return result, grid, st_lyrnorm_new
end

end
