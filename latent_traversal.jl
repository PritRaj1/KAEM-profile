using ConfParser, Random, JLD2, ComponentArrays, Lux, Reactant, Statistics
using CairoMakie, LaTeXStrings, Colors

ENV["DEVICE"] = "cpu"

CairoMakie.activate!(type = "png")

dataset = "CELEBA"
file_loc = "logs/Vanilla/CELEBA/ULA/mixture/"
save_dir = file_loc * "traversals/"
mkpath(save_dir)

num_steps = 10
num_prior_samples = 500
num_base_samples = 3
num_top_dims = 10

conf = ConfParse("config/celeba_config.ini")
parse_conf!(conf)
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")

ENV["DEVICE"] = retrieve(conf, "TRAINING", "device")
ENV["PERCEPTUAL"] = retrieve(conf, "TRAINING", "use_perceptual_loss")

include("src/utils.jl")
using .Utils

include("src/pipeline/trainer.jl")
using .trainer
using .trainer: seed_rand
using .trainer.KAEM_model: load_params

# Load model
saved_data = load(file_loc * "saved_model.jld2")
rng = Random.MersenneTwister(1)
t = init_trainer(rng, conf, dataset; img_resize = (64, 64), file_loc = file_loc, save_model = false)

t.ps = load_params(saved_data) |> pu
t.st_kan = saved_data["kan_state"] |> pu
t.st_lux = saved_data["lux_state"] |> pu

model = t.model
ps = t.ps
st_kan = t.st_kan
st_lux = Lux.testmode(t.st_lux)
q_size = model.prior.q_size
batch_size = model.batch_size

# Sample from prior to get empirical distribution
num_batches = num_prior_samples ÷ batch_size
z_all = zeros(Float32, q_size, 1, num_batches * batch_size)

for i in 1:num_batches
    st_rng = seed_rand(model; rng = rng)
    z, _ = Reactant.@jit model.sample_prior(model, ps, st_kan, st_lux, st_rng)
    idx = ((i - 1) * batch_size + 1):(i * batch_size)
    z_all[:, :, idx] = Array(z)
end

println("Collected $(size(z_all, 3)) prior samples")

# Per-dimension percentile ranges
quantile_vals = range(0.05, 0.95; length = num_steps)
z_percentiles = zeros(Float32, q_size, num_steps)

for d in 1:q_size
    vals = sort(vec(z_all[d, 1, :]))
    for (qi, qv) in enumerate(quantile_vals)
        idx = clamp(round(Int, qv * length(vals)), 1, length(vals))
        z_percentiles[d, qi] = vals[idx]
    end
end

# Compile
function decode(ps_gen, st_kan_gen, st_lux_gen, z)
    x̂, _ = model.lkhood.generator(ps_gen, st_kan_gen, st_lux_gen, z)
    return model.lkhood.output_activation(x̂)
end

z_dummy = pu(zeros(Float32, q_size, 1, batch_size))
decode_compiled = Reactant.@compile decode(ps.gen, st_kan.gen, st_lux.gen, z_dummy)
println("Decoder compiled.")

# Traverse each dim, measure variation, and save grids
for base_idx in 1:num_base_samples
    z_anchor = z_all[:, :, base_idx:base_idx]
    variation = zeros(Float32, q_size)
    decoded_per_dim = Dict{Int, Array{Float32}}()

    for dim in 1:q_size
        z_batch = repeat(z_anchor, 1, 1, batch_size)
        for qi in 1:num_steps
            z_batch[dim, 1, qi] = z_percentiles[dim, qi]
        end

        x_decoded = Array(decode_compiled(ps.gen, st_kan.gen, st_lux.gen, pu(z_batch)))
        variation[dim] = mean(std(x_decoded[:, :, :, 1:num_steps]; dims = 4))
        decoded_per_dim[dim] = x_decoded
    end

    top_dims = sortperm(variation; rev = true)[1:min(num_top_dims, q_size)]
    println("Base $base_idx — top dims: $top_dims")

    cell_size = 64
    label_col_width = 40
    header_row_height = 20
    fig_w = label_col_width + num_steps * cell_size + (num_steps - 1) * 2
    fig_h = header_row_height + num_top_dims * cell_size + (num_top_dims - 1) * 2

    fig = Figure(
        size = (fig_w, fig_h),
        fontsize = 10,
        backgroundcolor = :white,
        figure_padding = (4, 4, 4, 4),
    )

    # Image grid
    for (row, dim) in enumerate(top_dims)
        x_decoded = decoded_per_dim[dim]
        for qi in 1:num_steps
            ax = CairoMakie.Axis(
                fig[row + 1, qi + 1],
                aspect = DataAspect(),
                width = Fixed(cell_size),
                height = Fixed(cell_size),
            )
            hidedecorations!(ax)
            hidespines!(ax)
            raw = clamp.(x_decoded[:, :, :, qi], 0.0f0, 1.0f0)
            img = permutedims(raw, (2, 1, 3))
            rgb = RGB.(img[:, :, 1], img[:, :, 2], img[:, :, 3])
            image!(ax, rgb)
        end
    end

    # Row labels (latent dimension)
    for (row, dim) in enumerate(top_dims)
        Label(fig[row + 1, 1], L"z_{%$dim}", fontsize = 10, halign = :right)
    end

    # Column headers (quantile values)
    for qi in 1:num_steps
        qv = round(quantile_vals[qi]; digits = 2)
        Label(fig[1, qi + 1], "$qv", fontsize = 8, valign = :bottom)
    end

    colgap!(fig.layout, 2)
    rowgap!(fig.layout, 2)
    colsize!(fig.layout, 1, Fixed(label_col_width))
    rowsize!(fig.layout, 1, Fixed(header_row_height))

    save(save_dir * "traversal_base$(base_idx).png", fig, px_per_unit = 3)
    println("Saved traversal_base$(base_idx).png")
end

println("Results in $save_dir")
