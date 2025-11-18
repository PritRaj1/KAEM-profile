#!/usr/bin/env julia

ENV["GPU"] = true

println("Loading packages...")
using Reactant, ConfParser, Lux, ComponentArrays, Random

# Get the directory where this script is located
const SCRIPT_DIR = @__DIR__
const PROJECT_ROOT = dirname(SCRIPT_DIR)

println("Loading Utils...")
include(joinpath(PROJECT_ROOT, "src/utils.jl"))
using .Utils

println("Loading KAEM...")
include(joinpath(PROJECT_ROOT, "src/KAEM/KAEM.jl"))
using .KAEM_model

println("Loading ModelSetup...")
include(joinpath(PROJECT_ROOT, "src/KAEM/model_setup.jl"))
using .ModelSetup

# Load config
println("\nLoading config...")
conf = ConfParse(joinpath(SCRIPT_DIR, "test_conf.ini"))
parse_conf!(conf)

Random.seed!(42)

println("\n=== Model Configuration ===")
println("Layer widths: ", retrieve(conf, "EbmModel", "layer_widths"))
println("Grid size: ", retrieve(conf, "EbmModel", "grid_size"))
println("Spline function: ", retrieve(conf, "EbmModel", "spline_function"))

println("\n=== Initializing Model ===")
b_size = 8
x_shape = (32, 32, 1)
dataset = randn(Float32, x_shape..., b_size * 10)
@time model = init_KAEM(dataset, conf, x_shape)
println("Model depth: ", model.prior.depth)
println("Number of functions: ", length(model.prior.fcns_qp))

x_test = first(model.train_loader) |> pu
println("\n=== Preparing Model ===")
@time model, ps, st_kan, st_lux = prep_model(model, x_test, MLIR = false)

println("\n=== Creating Test Inputs ===")
println("Compiling sampling function...")
@time sample_prior_compiled = Reactant.@compile model.sample_prior(model, b_size, ps, st_kan, st_lux, Random.default_rng())
println("Sampling from prior...")
@time z_test = first(sample_prior_compiled(model, b_size, ps, st_kan, st_lux, Random.default_rng()))
println("Input shape: ", size(z_test))

println("\n=== Generating HLO Code ===")
println("Analyzing generated code size...")
@time begin
    mktemp() do path, io
        redirect_stdout(io) do
            Reactant.@code_hlo model.log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm)
        end
        close(io)
        hlo_str = read(path, String)
        hlo_lines = count(==('\n'), hlo_str)
        hlo_bytes = sizeof(hlo_str)

        println("\n=== HLO Statistics ===")
        println("  Lines: $hlo_lines")
        println("  Size: $(round(hlo_bytes / 1024^2, digits = 2)) MB")

        if hlo_lines > 50000
            println("\n⚠️  CRITICAL: Extremely large HLO ($hlo_lines lines)!")
            println("    This will take forever to compile.")
        elseif hlo_lines > 10000
            println("\n⚠️  WARNING: Very large HLO ($hlo_lines lines)")
            println("    Compilation will be slow (10+ minutes)")
        elseif hlo_lines > 1000
            println("\n⚠️  Large HLO ($hlo_lines lines)")
            println("    Compilation may take a few minutes")
        else
            println("\n✓  Reasonable HLO size")
        end
    end
end

println("\n=== Attempting Compilation ===")
println("This may take a while depending on HLO size...")
println("Press Ctrl+C to abort if it hangs.")
@time begin
    try
        compiled_fn = Reactant.@compile model.log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm)
        println("✓ Compilation successful!")

        println("\n=== Running Compiled Function ===")
        @time result_compiled = compiled_fn(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm)
        println("Result shape: ", size(result_compiled[1]))
        println("Result type: ", typeof(result_compiled[1]))
    catch e
        println("\n✗ Compilation failed: ", e)
    end
end

println("\n✓ Success! Compilation complete.")
println("Note: Subsequent calls will be much faster (cached).")
