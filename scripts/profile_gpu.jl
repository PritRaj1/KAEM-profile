using Random, ConfParser, Lux, Reactant, Enzyme, ComponentArrays, Accessors, Statistics, JSON

include("src/utils.jl")
using .Utils

include("src/KAEM/KAEM.jl")
using .KAEM_model
using .KAEM_model.EBM_Model
using .KAEM_model.GeneratorModel

include("src/KAEM/model_setup.jl")
using .ModelSetup

include("src/pipeline/optimizer.jl")
include("src/pipeline/data_utils.jl")
using .optimization
using .DataUtils: get_vision_dataset

include("src/KAEM/rng.jl")
using .HLOrng

# --- Config ---
ENV["PERCEPTUAL"] = "false"
ENV["THERMO"] = "false"

gpu_name = try
    strip(readchomp(`nvidia-smi --query-gpu=name --format=csv,noheader`))
catch
    "unknown"
end
println("GPU: $gpu_name")

REPORT_DIR = joinpath("profiling_reports", "gpu")
TRACE_DIR = joinpath(REPORT_DIR, "traces")
HLO_DIR = joinpath(REPORT_DIR, "hlo")
mkpath(TRACE_DIR)
mkpath(HLO_DIR)

open(joinpath(REPORT_DIR, "device_info.txt"), "w") do f
    println(f, "device: gpu")
    println(f, "device_name: $gpu_name")
end

# --- Build model ---
rng = Random.MersenneTwister(1)
conf = ConfParse("config/nist_config.ini")
parse_conf!(conf)
ENV["DEVICE"] = retrieve(conf, "TRAINING", "device")

dataset, x_shape, _ = get_vision_dataset("MNIST", 1000, 100, 100; img_resize = (32, 32), cnn = false)
model = init_KAEM(dataset, conf, x_shape; rng = rng)
x_batch, _ = iterate(model.train_loader)
x_batch = pu(x_batch)
opt = create_opt(conf)
model, opt_st, ps, sk, sl, sr = prep_model(model, x_batch, opt; rng = rng, MLIR = false)
sr = seed_rand(model; rng = rng)

ebm = model.prior
gen = model.lkhood
z = pu(randn(Float32, ebm.q_size, ebm.p_size, model.batch_size))

# ============================================================
# 1. Per-component XLA traces + timings
# ============================================================
println("\n" * "="^60)
println("SECTION 1: Per-component XLA traces")
println("="^60)

function profile_component(label, f, args...; n_trace = 30, n_time = 20)
    println("\n--- $label ---")
    compiled = Reactant.@compile f(args...)
    compiled(args...)  # warmup

    dir = joinpath(TRACE_DIR, label)
    mkpath(dir)
    Reactant.Profiler.with_profiler(dir) do
        for _ in 1:n_trace
            compiled(args...)
        end
    end

    times = [(@elapsed compiled(args...)) for _ in 1:n_time]
    med = median(times) * 1000
    println("  $(round(med, digits = 3)) ms")
    return Dict("median_ms" => med, "min_ms" => minimum(times) * 1000, "max_ms" => maximum(times) * 1000)
end

results = Dict{String, Any}()

ebm_fwd(z, p, s, l) = ebm(p, s, l, z)
results["ebm_forward"] = profile_component("ebm_forward", ebm_fwd, z, ps.ebm, sk.ebm, sl.ebm)

gen_fwd(p, s, l, z) = gen.output_activation(first(gen.generator(p, s, l, z)))
results["gen_forward"] = profile_component("gen_forward", gen_fwd, ps.gen, sk.gen, sl.gen, z)

quad_fwd(p, s, l, q) = ebm.quad(ebm, p, s, l, q)
results["quadrature"] = profile_component("quadrature", quad_fwd, ps.ebm, sk.ebm, sl.ebm, sk.quad)

its_fwd(m, p, s, l, r) = m.sample_prior(m, p, s, l, r)
results["inv_transform_sampling"] = profile_component("inv_transform_sampling", its_fwd, model, ps, sk, sl, sr)

z_samp = pu(Float32.(Array(first(model.sample_prior(model, ps, sk, sl, sr)))))
lp_fwd(z, p, s, l, q) = model.log_prior(z, ebm, p, s, l, q)
results["log_prior"] = profile_component("log_prior", lp_fwd, z_samp, ps.ebm, sk.ebm, sl.ebm, sk.quad)

results["generation"] = profile_component("generation", model, ps, sk, Lux.testmode(sl), sr; n_trace = 50)

results["full_train_step"] = profile_component(
    "full_train_step",
    model.train_step, opt_st, ps, sk, sl, x_batch, 1, sr; n_trace = 30, n_time = 30
)

open(joinpath(REPORT_DIR, "component_times.json"), "w") do f
    write(f, JSON.json(results, 2))
end
println("\nTraces saved to $TRACE_DIR")

# ============================================================
# 2. HLO IR per component
# ============================================================
println("\n" * "="^60)
println("SECTION 2: HLO IR")
println("="^60)

targets = [
    ("ebm_forward", () -> @code_hlo ebm(ps.ebm, sk.ebm, sl.ebm, z)),
    ("gen_forward", () -> @code_hlo gen.generator(ps.gen, sk.gen, sl.gen, z)),
    ("quadrature", () -> @code_hlo ebm.quad(ebm, ps.ebm, sk.ebm, sl.ebm, sk.quad)),
    ("inv_transform_sampling", () -> @code_hlo model.sample_prior(model, ps, sk, sl, sr)),
    ("log_prior", () -> @code_hlo model.log_prior(z, ebm, ps.ebm, sk.ebm, sl.ebm, sk.quad)),
    ("generation", () -> @code_hlo model(ps, sk, Lux.testmode(sl), sr)),
    ("train_step", () -> @code_hlo model.train_step(opt_st, ps, sk, sl, x_batch, 1, sr)),
]

for (label, get_hlo) in targets
    print("$label... ")
    try
        hlo_str = string(get_hlo())
        open(joinpath(HLO_DIR, "$label.txt"), "w") do f
            write(f, hlo_str)
        end
        println("ok")
    catch e
        println("FAILED: ", e)
    end
end
println("HLO saved to $HLO_DIR")

# ============================================================
# 3. Per-mode comparison (vanilla / variational / thermo)
# ============================================================
println("\n" * "="^60)
println("SECTION 3: Per-mode comparison")
println("="^60)

include("src/pipeline/trainer.jl")
using .trainer

modes = [
    ("vanilla", "false", "0", "false", "false"),
    ("variational", "true", "0", "false", "false"),
    ("thermo", "false", "4", "true", "true"),
]

summary = Dict{String, Any}()

for (label, variational, n_t, use_thermo, use_langevin) in modes
    println("\n=== $label ===")
    try
        conf = ConfParse("config/nist_config.ini")
        parse_conf!(conf)
        ENV["DEVICE"] = retrieve(conf, "TRAINING", "device")
        ENV["THERMO"] = use_thermo
        commit!(conf, "VARIATIONAL", "use_variational", variational)
        commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", n_t)
        commit!(conf, "POST_LANGEVIN", "use_langevin", use_langevin)

        compile_t = @elapsed begin
            t = init_trainer(rng, conf, "MNIST"; img_resize = (32, 32), save_model = false)
        end

        t.st_rng = seed_rand(t.model; rng = t.rng)
        t.model.train_step(t.opt_state, t.ps, t.st_kan, t.st_lux, t.x, 1, t.st_rng)

        mkpath(joinpath(TRACE_DIR, "mode_$label"))
        Reactant.Profiler.with_profiler(joinpath(TRACE_DIR, "mode_$label")) do
            for _ in 1:30
                t.st_rng = seed_rand(t.model; rng = t.rng)
                t.model.train_step(t.opt_state, t.ps, t.st_kan, t.st_lux, t.x, 1, t.st_rng)
            end
        end

        times = Float64[]
        for _ in 1:20
            t.st_rng = seed_rand(t.model; rng = t.rng)
            push!(times, @elapsed t.model.train_step(t.opt_state, t.ps, t.st_kan, t.st_lux, t.x, 1, t.st_rng))
        end

        summary[label] = Dict(
            "compile_s" => compile_t,
            "median_ms" => median(times) * 1000,
            "min_ms" => minimum(times) * 1000,
            "max_ms" => maximum(times) * 1000,
        )
        println("  compile: $(round(compile_t, digits = 1))s, median: $(round(median(times) * 1000, digits = 2))ms")
    catch e
        println("  FAILED: ", e)
        summary[label] = Dict("error" => string(e))
    end
end

open(joinpath(REPORT_DIR, "mode_summary.json"), "w") do f
    write(f, JSON.json(summary, 2))
end

# ============================================================
# Summary
# ============================================================
println("\n" * "="^60)
println("DONE")
println("="^60)
println("  Traces:  $TRACE_DIR")
println("  HLO IR:  $HLO_DIR")
println("  Timings: $(joinpath(REPORT_DIR, "component_times.json"))")
println("  Modes:   $(joinpath(REPORT_DIR, "mode_summary.json"))")
println("\nOpen .pb traces at https://ui.perfetto.dev")
