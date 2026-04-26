using BenchmarkTools, ConfParser, Lux, Random, ComponentArrays, CSV, DataFrames, Reactant

ENV["DEVICE"] = "gpu"

include("../src/pipeline/data_utils.jl")
using .DataUtils: get_vision_dataset

include("../src/utils.jl")
using .Utils

include("../src/KAEM/KAEM.jl")
using .KAEM_model

include("../src/KAEM/model_setup.jl")
using .ModelSetup

include("../src/pipeline/optimizer.jl")
using .optimization

conf = ConfParse("config/svhn_config.ini")
parse_conf!(conf)
optimizer = create_opt(conf)

rng = Random.MersenneTwister(1)

commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")
commit!(conf, "CNN", "use_cnn_lkhood", "true")
commit!(conf, "SEQ", "sequence_length", "0")
commit!(conf, "TRAINING", "verbose", "false")

dataset, img_size = get_vision_dataset(
    "SVHN",
    parse(Int, retrieve(conf, "TRAINING", "N_train")),
    parse(Int, retrieve(conf, "TRAINING", "N_test")),
    parse(Int, retrieve(conf, "TRAINING", "num_generated_samples"));
    cnn = true,
)[1:2]

function setup_model(n_z)
    commit!(conf, "EbmModel", "layer_widths", "$(n_z), $(2 * n_z + 1)")
    commit!(conf, "GeneratorModel", "widths", "$(2 * n_z + 1), $(4 * n_z + 2)")

    model = init_KAEM(dataset, conf, img_size; rng = rng)
    x_test, loader_state = iterate(model.train_loader)
    x_test = pu(x_test)
    model, opt_state, params, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; rng = rng, MLIR = false)

    return model, opt_state, params, st_kan, st_lux, st_rng, x_test
end

function benchmark_latent_dim(
        opt_state,
        params,
        st_kan,
        st_lux,
        st_rng,
        model,
        x_test
    )
    return model.train_step(
        opt_state,
        params,
        st_kan,
        st_lux,
        x_test,
        1,
        st_rng,
    )
end

for (sampler, output_name) in [
        ("importance", "latent_dim_importance.csv"),
        ("ula", "latent_dim_ula.csv"),
    ]
    commit!(conf, "POST_LANGEVIN", "sampler", sampler)
    println("=== Benchmarking KAEM training with sampler = $(sampler) ===")

    results = DataFrame(
        n_z = Int[],
        time_mean = Float64[],
        time_std = Float64[],
        memory_estimate = Float64[],
        allocations = Int[],
        gc_percent = Float64[],
    )

    for n_z in [10, 20, 30, 40, 50]
        println("Benchmarking n_z = $n_z (sampler = $(sampler))...")

        model, opt_state, params, st_kan, st_lux, st_rng, x_test = setup_model(n_z)

        b = @benchmark begin
            result = f(
                $opt_state,
                $params,
                $st_kan,
                $st_lux,
                $st_rng,
                $model,
                $x_test
            )
            Reactant.synchronize(result)
        end setup = (
            f = Reactant.@compile sync = true benchmark_latent_dim(
                $opt_state,
                $params,
                $st_kan,
                $st_lux,
                $st_rng,
                $model,
                $x_test
            )
        )

        push!(
            results,
            (
                n_z,
                b.times[end] / 1.0e9,
                std(b.times) / 1.0e9,
                b.memory / (1024^3),
                b.allocs,
                b.gctimes[end] / b.times[end] * 100,
            ),
        )
    end

    CSV.write("benches/results/$(output_name)", results)
    println("Results saved to $(output_name)")
    println(results)
end
