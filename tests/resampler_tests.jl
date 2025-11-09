# using Test, Random, LinearAlgebra
# using NNlib: softmax

# ENV["GPU"] = false

# include("../src/utils.jl")
# using .Utils

# include("../src/KAEM/gen/resamplers.jl")
# using .WeightResamplers

# function test_systematic_resampler()
#     Random.seed!(42)
#     weights = rand(Float32, 4, 6) |> pu
#     ESS_bool = rand(Bool, 4) |> pu
#     r = SystematicResampler(0.5, true)

#     idxs = r(softmax(weights; dims = 2))
#     @test size(idxs) == (4, 6)
#     return @test !any(isnan, idxs)
# end

# function test_stratified_resampler()
#     Random.seed!(42)
#     weights = rand(Float32, 4, 4) |> pu
#     ESS_bool = rand(Bool, 4) |> pu
#     r = StratifiedResampler(0.5, true)

#     idxs = r(softmax(weights; dims = 2))
#     @test size(idxs) == (4, 4)
#     return @test !any(isnan, idxs)
# end

# function test_residual_resampler()
#     Random.seed!(42)
#     weights = rand(Float32, 4, 4) |> pu
#     ESS_bool = rand(Bool, 4) |> pu
#     r = ResidualResampler(0.5, true)

#     idxs = r(softmax(weights; dims = 2))
#     @test size(idxs) == (4, 4)
#     return @test !any(isnan, idxs)
# end

# @testset "Resampler Tests" begin
#     test_systematic_resampler()
#     test_stratified_resampler()
#     test_residual_resampler()
# end
