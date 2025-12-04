using Test, Random, LinearAlgebra, Lux, ConfParser, ComponentArrays

ENV["THERMO"] = "true"
ENV["GPU"] = true

include("../src/utils.jl")
using .Utils

include("../src/KAEM/KAEM.jl")
using .KAEM_model

include("../src/KAEM/model_setup.jl")
using .ModelSetup

include("../src/pipeline/optimizer.jl")
using .optimization

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "4")
out_dim = parse(Int, retrieve(conf, "GeneratorModel", "output_dim"))
optimizer = create_opt(conf)

function test_model_derivative()
    Random.seed!(42)
    dataset = randn(Float32, 32, 32, 1, 50)
    model = init_KAEM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> pu
    model, opt_state, ps, st_kan, st_lux = prep_model(model, x_test, optimizer)
    swap_replica_idxs = rand(1:(model.N_t - 1), model.posterior_sampler.N) |> pu

    loss, grads, st_ebm, st_gen =
        model.loss_func(ps, st_kan, st_lux, x_test, 1, Random.default_rng(), swap_replica_idxs)

    grads = Array(grads)
    @test all(iszero, grads)
    return @test !any(isnan, grads)
end

@testset "Thermodynamic Integration Tests" begin
    test_model_derivative()
end
