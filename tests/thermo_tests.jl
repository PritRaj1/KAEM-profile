using Test, Random, LinearAlgebra, Lux, ConfParser, ComponentArrays

ENV["THERMO"] = "true"
ENV["GPU"] = true

include("../src/utils.jl")
using .Utils

include("../src/KAEM/KAEM.jl")
using .KAEM_model

include("../src/KAEM/model_setup.jl")
using .ModelSetup

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "4")
out_dim = parse(Int, retrieve(conf, "GeneratorModel", "output_dim"))

function test_model_derivative()
    Random.seed!(42)
    dataset = randn(Float32, 32, 32, 1, 50)
    model = init_KAEM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> pu
    model, ps, st_kan, st_lux = prep_model(model, x_test)
    swap_replica_idxs = rand(1:(model.N_t - 1), model.posterior_sampler.N) |> pu

    ps_before = Array(ps)
    loss, ps, _, st_ebm, st_gen =
        model.train_step(opt_state, ps, st_kan, st_lux, x_test, 1, Random.default_rng(), nothing)

    ps_after = Array(ps)
    @test any(ps_before .!= ps_after)
    return @test !any(isnan, ps)
end

@testset "Thermodynamic Integration Tests" begin
    test_model_derivative()
end
