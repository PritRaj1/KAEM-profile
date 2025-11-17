#!/usr/bin/env julia

ENV["GPU"] = true

println("Loading packages...")
using Reactant, ConfParser, Lux, ComponentArrays, Random

println("Loading Utils...")
include("src/utils.jl")
using .Utils

println("Loading KAEM...")
include("src/KAEM/KAEM.jl")
using .KAEM_model

println("Loading ModelSetup...")
include("src/KAEM/model_setup.jl")
using .ModelSetup

# Load config
println("\nLoading config...")
conf = ConfParse("tests/test_conf.ini")
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
# Sample from the prior to get correct shape
println("Sampling from prior...")
@time z_test = first(model.sample_prior(model, b_size, ps, st_kan, st_lux, Random.default_rng()))
println("Input shape: ", size(z_test))

println("\n=== Testing Forward Pass (Uncompiled) ===")
@time result_uncompiled = model.log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm)
println("Result shape: ", size(result_uncompiled[1]))
println("Result type: ", typeof(result_uncompiled[1]))

println("\n=== Generating HLO Code (Tracing Only) ===")
println("This shows the size of generated code WITHOUT compilation:")
@time begin
    hlo_str = sprint() do io
        redirect_stdout(io) do
            Reactant.@code_hlo model.log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm)
        end
    end
    hlo_lines = count(==('\n'), hlo_str)
    println("Generated HLO has $hlo_lines lines")
    if hlo_lines > 10000
        println("⚠️  WARNING: Very large HLO code ($hlo_lines lines) - compilation will be slow!")
    elseif hlo_lines > 1000
        println("⚠️  Large HLO code ($hlo_lines lines) - compilation may take a while")
    else
        println("✓  Reasonable HLO size")
    end
end

println("\n=== Compiling (This is the slow part) ===")
println("If this hangs, the HLO code is too large.")
println("Try reducing layer_widths or grid_size in your config.")
@time compiled_fn = Reactant.@compile model.log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm)

println("\n=== Running Compiled Function ===")
@time result_compiled = compiled_fn(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm)

println("\n✓ Success! Compilation complete.")
println("Note: Subsequent calls will be much faster (cached).")
