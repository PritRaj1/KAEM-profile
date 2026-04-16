using ConfParser, Random, JLD2, ComponentArrays, Lux, Reactant, Statistics, LinearAlgebra
using CairoMakie, LaTeXStrings, Colors

ENV["DEVICE"] = "cpu"
CairoMakie.activate!(type = "png")

dataset = length(ARGS) >= 1 ? ARGS[1] : "CELEBA"
mode = length(ARGS) >= 2 ? ARGS[2] : "vanilla"

dataset_configs = Dict(
    "CELEBA" => (config = "config/celeba_config.ini", resize = (64, 64)),
    "SVHN" => (config = "config/svhn_config.ini", resize = (32, 32)),
)

haskey(dataset_configs, dataset) || error("Unknown dataset: $dataset. Use one of: $(keys(dataset_configs))")
ds = dataset_configs[dataset]

file_loc = (
    mode == "thermo" ? "logs/Thermodynamic/$(dataset)/ULA/mixture/" :
        "logs/Vanilla/$(dataset)/ULA/mixture/"
)
save_dir = "figures/interpolations/$(dataset)/$(mode)/"
mkpath(save_dir)

num_interp_steps = 8
num_pairs = 6
num_prior_samples = 500

conf = ConfParse(ds.config)
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
t = init_trainer(rng, conf, dataset; img_resize = ds.resize, file_loc = file_loc, save_model = false)

t.ps = load_params(saved_data) |> pu
t.st_kan = saved_data["kan_state"] |> pu
t.st_lux = saved_data["lux_state"] |> pu

model = t.model
ps = t.ps
st_kan = t.st_kan
st_lux = Lux.testmode(t.st_lux)
q_size = model.prior.q_size
batch_size = model.batch_size

# Sample from prior
num_batches = num_prior_samples ÷ batch_size
z_all = zeros(Float32, q_size, 1, num_batches * batch_size)
for i in 1:num_batches
    st_rng = seed_rand(model; rng = rng)
    z, _ = Reactant.@jit model.sample_prior(model, ps, st_kan, st_lux, st_rng)
    idx = ((i - 1) * batch_size + 1):(i * batch_size)
    z_all[:, :, idx] = Array(z)
end
println("Sampled $(size(z_all, 3)) latent vectors")

# Compile decoder
function decode(ps_gen, st_kan_gen, st_lux_gen, z)
    x̂, _ = model.lkhood.generator(ps_gen, st_kan_gen, st_lux_gen, z)
    return model.lkhood.output_activation(x̂)
end

z_dummy = pu(zeros(Float32, q_size, 1, batch_size))
decode_compiled = Reactant.@compile decode(ps.gen, st_kan.gen, st_lux.gen, z_dummy)
println("Decoder compiled.")

# SLERP: spherical linear interpolation
function slerp(z1, z2, t)
    z1_flat = vec(z1)
    z2_flat = vec(z2)
    n1 = z1_flat / max(norm(z1_flat), eps(Float32))
    n2 = z2_flat / max(norm(z2_flat), eps(Float32))
    omega = acos(clamp(dot(n1, n2), -1.0f0, 1.0f0))
    if omega < 1.0f-6
        return (1.0f0 - t) .* z1 .+ t .* z2
    end
    s = sin(omega)
    return (sin((1.0f0 - t) * omega) / s) .* z1 .+ (sin(t * omega) / s) .* z2
end

# Select pairs far apart in latent space
N = size(z_all, 3)
pair_dists = Dict{Tuple{Int, Int}, Float32}()
for i in 1:N, j in (i + 1):N
    pair_dists[(i, j)] = sum((z_all[:, :, i] .- z_all[:, :, j]) .^ 2)
end
sorted_pairs = sort(collect(pair_dists); by = last, rev = true)

pairs = Tuple{Int, Int}[]
used = Set{Int}()
for (p, _) in sorted_pairs
    (p[1] in used || p[2] in used) && continue
    push!(pairs, p)
    push!(used, p[1], p[2])
    length(pairs) >= num_pairs && break
end
println("Selected pairs: $pairs")

# Total columns: z_a + num_interp_steps intermediates + z_b
num_cols = num_interp_steps + 2
ts = range(0.0f0, 1.0f0; length = num_cols)

# Decode all interpolations
all_rgb = Array{Matrix{RGB{Float32}}}(undef, num_pairs, num_cols)

for (row, (i, j)) in enumerate(pairs)
    z_a = z_all[:, :, i:i]
    z_b = z_all[:, :, j:j]

    z_batch = repeat(z_a, 1, 1, batch_size)
    for (ci, t_val) in enumerate(ts)
        z_batch[:, :, ci] .= slerp(z_a, z_b, t_val)
    end

    x_decoded = Array(decode_compiled(ps.gen, st_kan.gen, st_lux.gen, pu(z_batch)))
    for ci in 1:num_cols
        raw = clamp.(x_decoded[:, :, :, ci], 0.0f0, 1.0f0)
        rgb = RGB.(raw[:, :, 1], raw[:, :, 2], raw[:, :, 3])
        all_rgb[row, ci] = rot180(rgb)
    end
end

# Plot
cell = 48
gap = 2
row_gap = 6
header_h = 20
fig_w = num_cols * cell + (num_cols - 1) * gap
fig_h = header_h + num_pairs * cell + (num_pairs - 1) * row_gap

fig = Figure(size = (fig_w, fig_h), backgroundcolor = :white, figure_padding = (2, 2, 2, 2))

for row in 1:num_pairs, col in 1:num_cols
    ax = CairoMakie.Axis(
        fig[row + 1, col],
        aspect = DataAspect(),
        width = Fixed(cell),
        height = Fixed(cell),
    )
    hidedecorations!(ax)
    hidespines!(ax)
    image!(ax, all_rgb[row, col])
end

Label(fig[1, 1], L"z_A", fontsize = 14, halign = :center, valign = :bottom)
Label(fig[1, div(num_cols, 2):(div(num_cols, 2) + 1)], L"\longrightarrow", fontsize = 14, halign = :center, valign = :bottom)
Label(fig[1, num_cols], L"z_B", fontsize = 14, halign = :center, valign = :bottom)

colgap!(fig.layout, gap)
rowgap!(fig.layout, row_gap)
rowsize!(fig.layout, 1, Fixed(header_h))

save(save_dir * "slerp.png", fig, px_per_unit = 5)
println("Saved to $(save_dir)slerp.png")
