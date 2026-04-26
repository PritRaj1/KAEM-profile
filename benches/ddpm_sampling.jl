using BenchmarkTools, ConfParser, Lux, Random, ComponentArrays, CSV, DataFrames, Reactant

ENV["DEVICE"] = "gpu"

include("../src/baseline/training/trainer.jl")
using .Baseline: Utils, DDPMModel, DDPMSampling, BaselineRNG, DataUtils
using .Utils: pu
using .DDPMModel: init_DDPM
using .DDPMSampling: sample_loop
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

function setup_ddpm_model()
    model = init_DDPM(conf, img_size; rng = rng)

    ps = Lux.initialparameters(rng, model)
    st = Lux.initialstates(rng, model)
    ps, st = ps |> ComponentArray |> Lux.f32 |> pu, st |> Lux.f32 |> pu

    st_rng = seed_rng(model; rng = rng, sampling = true)

    # Pre-resident the schedule tensors on device. Doing `|> pu` inside the
    # traced benchmark function confuses Reactant's @trace while loop, which
    # cannot wrap captured ConcretePJRTArrays in a RefValue.
    timesteps = model.sampling_timesteps |> pu
    alphas = model.sampling_alphas |> pu
    alphas_cumprod = model.sampling_alphas_cumprod |> pu
    betas = model.sampling_betas |> pu
    noise_masks = model.sampling_noise_masks |> pu
    step_masks = model.sampling_step_masks |> pu

    return (
        model.unet,
        ps,
        st,
        st_rng,
        timesteps,
        alphas,
        alphas_cumprod,
        betas,
        noise_masks,
        step_masks,
        model.sampling_num_steps,
    )
end

function benchmark_ddpm_sample(
        unet,
        ps,
        st,
        st_rng,
        timesteps,
        alphas,
        alphas_cumprod,
        betas,
        noise_masks,
        step_masks,
        num_steps,
    )
    return first(
        sample_loop(
            unet,
            ps,
            Lux.testmode(st),
            st_rng,
            timesteps,
            alphas,
            alphas_cumprod,
            betas,
            noise_masks,
            step_masks,
            num_steps,
        ),
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

# DDPM operates in pixel space and has no latent dimension.
println("Benchmarking DDPM sampling...")

(unet, ps, st, st_rng, timesteps, alphas, alphas_cumprod,
    betas, noise_masks, step_masks, num_steps) = setup_ddpm_model()

b = @benchmark begin
    result = f(
        $unet,
        $ps,
        $st,
        $st_rng,
        $timesteps,
        $alphas,
        $alphas_cumprod,
        $betas,
        $noise_masks,
        $step_masks,
        $num_steps,
    )
    Reactant.synchronize(result)
end setup = (
    f = Reactant.@compile sync = true benchmark_ddpm_sample(
        $unet,
        $ps,
        $st,
        $st_rng,
        $timesteps,
        $alphas,
        $alphas_cumprod,
        $betas,
        $noise_masks,
        $step_masks,
        $num_steps,
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

CSV.write("benches/results/ddpm_sampling.csv", results)
println("Results saved to ddpm_sampling.csv")
println(results)
