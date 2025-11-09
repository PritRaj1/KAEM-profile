# using Test, Random, LinearAlgebra, Lux, ConfParser, ComponentArrays


# ENV["GPU"] = false

# include("../src/utils.jl")
# using .Utils

# include("../src/KAEM/KAEM.jl")
# using .KAEM_model

# include("../src/KAEM/model_setup.jl")
# using .ModelSetup

# include("../src/KAEM/gen/loglikelihoods.jl")
# using .LogLikelihoods

# conf = ConfParse("tests/test_conf.ini")
# parse_conf!(conf)
# out_dim = parse(Int, retrieve(conf, "GeneratorModel", "output_dim"))
# b_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
# MC_sample_size = parse(Int, retrieve(conf, "TRAINING", "importance_sample_size"))
# z_dim = last(parse.(Int, retrieve(conf, "EbmModel", "layer_widths")))

# Random.seed!(42)

# function test_generate()
#     Random.seed!(42)
#     commit!(conf, "CNN", "use_cnn_lkhood", "false")
#     dataset = randn(Float32, 32, 32, 1, 50)
#     model = init_KAEM(dataset, conf, (32, 32, 1))
#     x_test = first(model.train_loader) |> pu
#     model, ps, st_kan, st_lux = prep_model(model, x_test)
#     ps = ps

#     z = first(model.sample_prior(model, b_size, ps, st_kan, st_lux, Random.default_rng()))
#     x, _ = model.lkhood.generator(ps.gen, st_kan.gen, st_lux.gen, z)
#     @test size(x) == (32, 32, 1, b_size)
#     return @test !any(isnan, x)
# end

# function test_logllhood()
#     Random.seed!(42)
#     dataset = randn(Float32, 32, 32, 1, 50)
#     model = init_KAEM(dataset, conf, (32, 32, 1))
#     x_test = first(model.train_loader) |> pu
#     model, ps, st_kan, st_lux = prep_model(model, x_test)
#     ps = ps

#     x = randn(Float32, 32, 32, 1, b_size) |> pu
#     z = first(model.sample_prior(model, b_size, ps, st_kan, st_lux, Random.default_rng()))
#     noise = randn(Float32, 32, 32, 1, b_size, b_size) |> pu
#     logllhood, _ =
#         log_likelihood_IS(z, x, model.lkhood, ps.gen, st_kan.gen, st_lux.gen, noise)
#     @test size(logllhood) == (b_size, b_size)
#     return @test !any(isnan, logllhood)
# end

# function test_cnn_generate()
#     Random.seed!(42)
#     commit!(conf, "CNN", "use_cnn_lkhood", "true")
#     dataset = randn(Float32, 32, 32, out_dim, 50)
#     model = init_KAEM(dataset, conf, (32, 32, out_dim))
#     x_test = first(model.train_loader) |> pu
#     model, ps, st_kan, st_lux = prep_model(model, x_test)
#     ps = ps

#     z = first(model.sample_prior(model, b_size, ps, st_kan, st_lux, Random.default_rng()))
#     x = first(model.lkhood.generator(ps.gen, st_kan.gen, st_lux.gen, z))
#     @test size(x) == (32, 32, out_dim, b_size)
#     @test !any(isnan, x)
#     return commit!(conf, "CNN", "use_cnn_lkhood", "false")
# end

# function test_seq_generate()
#     Random.seed!(42)
#     commit!(conf, "SEQ", "sequence_length", "8")

#     dataset = randn(Float32, out_dim, 8, 50)
#     model = init_KAEM(dataset, conf, (out_dim, 8))
#     x_test = first(model.train_loader) |> pu
#     model, ps, st_kan, st_lux = prep_model(model, x_test)
#     ps = ps

#     z = first(model.sample_prior(model, b_size, ps, st_kan, st_lux, Random.default_rng()))
#     x, _ = model.lkhood.generator(ps.gen, st_kan.gen, st_lux.gen, z)
#     @test size(x) == (out_dim, 8, b_size)
#     @test !any(isnan, x)
#     return commit!(conf, "SEQ", "sequence_length", "1")
# end

# @testset "KAN Likelihood Tests" begin
#     test_generate()
#     test_logllhood()
#     test_cnn_generate()
#     test_seq_generate()
# end
