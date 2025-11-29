using JLD2,
    Lux,
    ComponentArrays,
    ConfParser,
    LaTeXStrings,
    Makie,
    GLMakie,
    Random,
    Accessors

ENV["GPU"] = "false"

include("../src/utils.jl")
using .Utils

include("../src/KAEM/KAEM.jl")
using .KAEM_model

include("../src/pipeline/trainer.jl")
using .trainer


for fcn_type in ["RBF", "FFT"]
    for prior_type in ["gaussian", "lognormal", "uniform", "ebm"]
        for dataset_name in ["DARCY_FLOW", "MNIST", "FMNIST"]
            file = "../logs/Vanilla/$(dataset_name)/importance/$(prior_type)_$(fcn_type)/univariate/saved_model.jld2"

            conf_loc = Dict(
                "DARCY_FLOW" => "config/darcy_flow_config.ini",
                "MNIST" => "config/nist_config.ini",
                "FMNIST" => "config/nist_config.ini",
            )[dataset_name]

            conf = ConfParse(conf_loc)
            parse_conf!(conf)
            commit!(conf, "EbmModel", "π_0", prior_type)

            if fcn_type == "RBF"
                commit!(conf, "EbmModel", "spline_function", "RBF")
                commit!(conf, "EbmModel", "base_activation", "silu")
            else
                commit!(conf, "EbmModel", "spline_function", "FFT")
                commit!(conf, "EbmModel", "base_activation", "none")
            end

            saved_data = load(file)

            ps = saved_data["params"] .|> Float32
            st_kan = saved_data["kan_state"]
            st_lux = saved_data["lux_state"]

            rng = Random.MersenneTwister(1)
            t = init_trainer(rng, conf, dataset_name; file_loc = "garbage/")
            prior = t.model.prior

            ps = ps.ebm
            st_kan = st_kan.ebm
            st_lux = st_lux.ebm
            t = nothing
            a, b = st_kan[:a].grid[:, 1], st_kan[:a].grid[:, end]

            no_grid = (
                prior.fcns_qp[1].spline_string == "FFT" ||
                    prior.fcns_qp[1].spline_string == "Cheby"
            )

            if no_grid
                a .= a .* 0.0f0 .+ st_kan[:a].min
                b .= b .* 0.0f0 .+ st_kan[:a].max
            end

            z = (a + b) ./ 2 .+ (b - a) ./ 2 .* pu(prior.nodes)
            π_0 = prior.π_pdf(z[:, :, :], ps.dist.π_μ, ps.dist.π_σ)

            f = first(prior(ps, st_kan, st_lux, z))
            f = exp.(f) .* permutedims(π_0, (3, 1, 2))
            z, f, π_0 = z,
                softmax(f; dims = 3),
                softmax(π_0; dims = 2)

            # Components to plot (q, p)
            plot_components = [(1, 1), (1, 2), (1, 3)]
            colours = [:red, :blue, :green]

            mkpath("../figures/results/priors/$(dataset_name)")

            for (i, (q, p)) in enumerate(plot_components)
                fig = Makie.Figure(
                    size = (800, 800),
                    ffont = "Computer Modern",
                    fontsize = 20,
                    backgroundcolor = :white,
                    show_axis = false,
                    show_grid = false,
                    show_axis_labels = false,
                    show_legend = false,
                    show_colorbar = false,
                )
                ax = Makie.Axis(
                    fig[1, 1],
                    title = L"Prior component, ${\exp(f_{%$q,%$p}(z)) \cdot \pi_0(z)} \; / \; {\textbf{Z}_{%$q,%$p}}$",
                )

                band!(
                    ax,
                    z[p, :],
                    0 .* f[q, p, :],
                    f[q, p, :],
                    color = (colours[i], 0.3),
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
                y_min = minimum([minimum(f[q, p, :]), minimum(π_0[p, :])])
                ylims!(ax, y_min, nothing)
                axislegend(ax)
                hidedecorations!(ax)
                hidespines!(ax)
                save(
                    "../figures/results/priors/$(dataset_name)/$(prior_type)_$(fcn_type)_$(q)_$(p).png",
                    fig,
                )
            end
        end
    end
end
