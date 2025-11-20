using Test, Random, LinearAlgebra, Lux, ConfParser, ComponentArrays

ENV["THERMO"] = "true"
ENV["GPU"] = false

include("../src/utils.jl")
using .Utils

include("../src/KAEM/KAEM.jl")
using .KAEM_model

include("../src/KAEM/model_setup.jl")
using .ModelSetup

include("../src/KAEM/loss_fcns/thermodynamic.jl")
using .ThermodynamicIntegration: sample_thermo

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
    swap_replica_idxs = rand(rng, 1:(model.N_t - 1), model.posterior_sampler.N)


    loss, ∇, st_ebm, st_gen =
        model.loss_fcn(ps, st_kan, st_lux, model, x_test, 1, Random.default_rng(), swap_replica_idxs)

    ∇ = Array(∇)
    @test norm(∇) != 0
    return @test !any(isnan, ∇)
end

@testset "Thermodynamic Integration Tests" begin
    test_model_derivative()
end
