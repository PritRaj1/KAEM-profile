module PlotKAN

export plot_ebm!

using CarioMakie, LatexStrings
using NNlib: softmax

using ..Utils

include("../ebm/quadrature.jl")
using .Quadrature: get_gausslegendre

CairoMakie.activate!(type = "png")

function plot_ebm!(
        ebm,
        ps,
        st_kan,
        st_lux;
        plot_components = [(1, 1)],
        component_colours = [:red],
        file_loc = "../figures/results/priors/"
    )
    mkpath(file_loc)

    z = first(get_gausslegendre(ebm, st_kan))
    π_0 = prior.π_pdf(z, ps.dist.π_μ, ps.dist.π_σ)

    f = first(prior(ps, st_kan, st_lux, z))
    f = exp.(f) .* PermutedDimsArray(view(π_0, :, :, :), (3, 1, 2))
    z, f, π_0 = z,
        softmax(f; dims = 3),
        softmax(π_0; dims = 2)

    for (i, (q, p)) in enumerate(plot_components)
        fig = Figure(
            size = (800, 800),
            ffont = "Computer Modern",
            fontsize = 20,
            backgroundcolor = :white,
            show_axis = true,
            show_grid = true,
            show_axis_labels = true,
            show_legend = true,
            show_colorbar = false,
        )
        ax = Axis(
            fig[1, 1],
            title = "Prior component, q = $(q), p = $(p)",
            xlabel = L"z",
            ylabel = L"${\exp(f_{%$q,%$p}(z)) \cdot \pi_0(z)} \; / \; {\textbf{Z}_{%$q,%$p}}$",
        )

        band!(
            ax,
            z[p, :],
            0 .* f[q, p, :],
            f[q, p, :],
            color = (component_colours[i], 0.3),
            label = L"{\exp(f_{%$q,%$p}(z)) \cdot \pi_0(z)}",
        )
        lines!(ax, z[p, :], f[q, p, :], color = colours[i])
        band!(
            ax,
            z[p, :],
            0 .* f[q, p, :],
            π_0[p, :, 1],
            color = (:gray, 0.2),
            label = L"\pi_0(z)",
        )
        lines!(ax, z[p, :], π_0[p, :, 1], color = (:gray, 0.8))

        axislegend(ax)

        save(
            file_loc * "$(prior_type)_$(fcn_type)_$(q)_$(p).png",
            fig,
        )
    end
    return nothing
end

end
