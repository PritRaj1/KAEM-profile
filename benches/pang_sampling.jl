using BenchmarkTools, ConfParser, Lux, Random, ComponentArrays, CSV, DataFrames, Reactant

ENV["DEVICE"] = "gpu"

include("../src/baseline/training/trainer.jl")
using .Baseline: Utils, PangEBMModel, PangEBMSampling, BaselineRNG, DataUtils
using .Utils: pu
using .PangEBMModel: init_PangEBM
using .PangEBMSampling: generate_pang
using .BaselineRNG: seed_rng
using .DataUtils: get_vision_dataset

conf = ConfParse("config/baseline_svhn_config.ini")
parse_conf!(conf)

rng = Random.MersenneTwister(1)

commit!(conf, "TRAINING", "verbose", "false")

dataset, img_size = get_vision_dataset(
    "SVHN",
    parse(Int, retrieve(conf, "TRAINING", "N_train")),
    parse(Int, retrieve(conf, "TRAINING", "N_test")),
    parse(Int, retrieve(conf, "TRAINING", "num_generated_samples"));
    cnn = true,
)[1:2]

function setup_pang_model(latent_dim)
    commit!(conf, "PANG", "latent_dim", "$(latent_dim)")

    model = init_PangEBM(conf, img_size; rng = rng)

    ps = Lux.initialparameters(rng, model)
    st = Lux.initialstates(rng, model)
    ps, st = ps |> ComponentArray |> Lux.f32 |> pu, st |> Lux.f32 |> pu

    st_rng = seed_rng(model; rng = rng)

    return model, ps, st, st_rng
end

function benchmark_pang_sample(model, ps, st, st_rng)
    return first(
        generate_pang(model, ps, Lux.testmode(st), st_rng),
    )
end

results = DataFrame(
    latent_dim = Int[],
    time_mean = Float64[],
    time_std = Float64[],
    memory_estimate = Float64[],
    allocations = Int[],
    gc_percent = Float64[],
)

for latent_dim in [21, 41, 61, 81, 101]
    println("Benchmarking Pang sampling with latent_dim = $latent_dim...")

    model, ps, st, st_rng = setup_pang_model(latent_dim)

    b = @benchmark begin
        result = f(
            $model,
            $ps,
            $st,
            $st_rng,
        )
        Reactant.synchronize(result)
    end setup = (
        f = Reactant.@compile sync = true benchmark_pang_sample(
            $model,
            $ps,
            $st,
            $st_rng,
        )
    )

    push!(
        results,
        (
            latent_dim,
            b.times[end] / 1.0e9,
            std(b.times) / 1.0e9,
            b.memory / (1024^3),
            b.allocs,
            b.gctimes[end] / b.times[end] * 100,
        ),
    )
end

CSV.write("benches/results/pang_sampling.csv", results)
println("Results saved to pang_sampling.csv")
println(results)
