using BenchmarkTools, ConfParser, Lux, Random, ComponentArrays, CSV, DataFrames, Reactant, Optimisers, Flux

ENV["DEVICE"] = "gpu"

include("../src/baseline/training/trainer.jl")
using .Baseline: Utils, DDPMModel, DDPMLoss, BaselineRNG, optimization, DataUtils
using .Utils: pu
using .DDPMModel: init_DDPM
using .DDPMLoss: DDPMTrainStep
using .BaselineRNG: seed_rng
using .optimization: ManualAdam
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

function setup_ddpm_model()
    model = init_DDPM(conf, img_size; rng = rng)

    ps = Lux.initialparameters(rng, model)
    st = Lux.initialstates(rng, model)
    ps, st = ps |> ComponentArray |> Lux.f32 |> pu, st |> Lux.f32 |> pu

    lr = parse(Float32, retrieve(conf, "DDPM", "learning_rate"))
    opt = ManualAdam(lr)
    opt_state = Optimisers.setup(opt, ps)

    st_rng = seed_rng(model; rng = rng)

    train_loader = Flux.DataLoader(dataset; batchsize = model.batch_size, shuffle = true)
    x_test, _ = iterate(train_loader)
    x_test = pu(x_test)

    return model, opt_state, ps, st, st_rng, x_test
end

results = DataFrame(
    latent_dim = Int[],
    time_mean = Float64[],
    time_std = Float64[],
    memory_estimate = Float64[],
    allocations = Int[],
    gc_percent = Float64[],
)

function benchmark_ddpm_train(train_step, opt_state, ps, st, x, st_rng)
    return train_step(opt_state, ps, Lux.trainmode(st), x, st_rng)
end

# DDPM operates in pixel space and has no latent dimension.
println("Benchmarking DDPM training...")

model, opt_state, ps, st, st_rng, x_test = setup_ddpm_model()

train_step = DDPMTrainStep(model)

b = @benchmark begin
    result = f(
        $train_step,
        $opt_state,
        $ps,
        $st,
        $x_test,
        $st_rng,
    )
    Reactant.synchronize(result)
end setup = (
    f = Reactant.@compile sync = true benchmark_ddpm_train(
        $train_step,
        $opt_state,
        $ps,
        $st,
        $x_test,
        $st_rng,
    )
)

row = (
    b.times[end] / 1.0e9,
    std(b.times) / 1.0e9,
    b.memory / (1024^3),
    b.allocs,
    b.gctimes[end] / b.times[end] * 100,
)

for latent_dim in [21, 41, 61, 81, 101]
    push!(results, (latent_dim, row...))
end

CSV.write("benches/results/ddpm_latent_dim.csv", results)
println("Results saved to ddpm_latent_dim.csv")
println(results)
