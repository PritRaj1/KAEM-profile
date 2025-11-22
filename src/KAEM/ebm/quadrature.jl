module Quadrature

export GaussLegendreQuadrature, get_gausslegendre

using LinearAlgebra, Random, ComponentArrays, FastGaussQuadrature, Accessors

using ..Utils

negative_one = - ones(Float32, 1, 1, 1)

struct GaussLegendreQuadrature <: AbstractQuadrature end

function qfirst_exp_kernel(f, π0, Q, S)
    return exp.(f) .* reshape(π0, Q, 1, S)
end

function pfirst_exp_kernel(f, π0, P, S)
    return exp.(f) .* reshape(π0, 1, P, S)
end

function apply_mask(exp_fg, component_mask, Q, P, S)
    return dropdims(sum(reshape(exp_fg, Q, P, 1, S) .* component_mask; dims = 2); dims = 2)
end

function weight_kernel(trapz, weights, P, S)
    return reshape(weights, 1, P, S) .* trapz
end

function gauss_kernel(
        trapz,
        weights,
        Q,
        S
    )
    return reshape(weights, Q, 1, S) .* trapz
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
        a .= a .* 0.0f0 .+ st_kan[:a].min
        b .= b .* 0.0f0 .+ st_kan[:a].max
    end

    nodes, weights = gausslegendre(ebm.N_quad)
    nodes = ((a .+ b) ./ 2 .+ (b .- a) ./ 2) * nodes'
    weights = ((b .- a) ./ 2) * weights'
    return nodes, weights
end

function mix_return(nodes, π_nodes, weights, component_mask, Q, P, S)
    exp_fg = qfirst_exp_kernel(nodes, π_nodes, Q, S)
    trapz = apply_mask(exp_fg, component_mask, Q, P, S)
    trapz = gauss_kernel(trapz, weights, Q, S)
    return trapz
end

function univar_return(nodes, π_nodes, weights, Q, P, S)
    exp_fg = pfirst_exp_kernel(nodes, π_nodes, P, S)
    exp_fg = weight_kernel(exp_fg, weights, P, S)
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
    I, O = first(ebm.fcns_qp).in_dim, first(ebm.fcns_qp).out_dim
    Q, P, S = ebm.q_size, ebm.p_size, ebm.N_quad

    π_nodes = ebm.π_pdf(reshape(nodes, I, S, 1), ps.dist.π_μ, ps.dist.π_σ)
    π_nodes =
        ebm.prior_type == "learnable_gaussian" ? dropdims(π_nodes, dims = 3)' :
        dropdims(π_nodes, dims = 3)

    for i in 1:ebm.depth
        @reset ebm.fcns_qp[i].basis_function.S = S
    end

    @reset ebm.s_size = S

    # Energy function of each component
    nodes, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, nodes)

    # Choose component if mixture model else use all
    result = (
        mix_bool ?
            mix_return(nodes, π_nodes, weights, component_mask, Q, P, S) :
            univar_return(nodes, π_nodes, weights, Q, P, S)
    )

    return result, st_quad.nodes, st_lyrnorm_new
end

end
