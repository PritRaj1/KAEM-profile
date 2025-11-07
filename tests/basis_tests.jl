using Test, Random, LinearAlgebra, CUDA

ENV["GPU"] = true

include("../src/utils.jl")
using .Utils

include("../src/KAEM/kan/spline_bases.jl")
using .spline_functions

b, i, g, o, degree, σ = 5, 8, 7, 2, 2, pu([one(Float32)])

function test_extend_grid()
    Random.seed!(42)
    grid = rand(Float32, i, g) |> pu
    extended_grid = extend_grid(grid; k_extend = degree)
    return @test size(extended_grid, 2) == size(grid, 2) + 2 * degree
end

function test_B_spline_basis()
    Random.seed!(42)
    x_eval = rand(Float32, i, b) |> pu
    Random.seed!(42)
    grid = rand(Float32, i, g) |> pu
    extended_grid = extend_grid(grid; k_extend = degree)
    coef = rand(Float32, i, o, g + degree - 1) |> pu

    basis_function = B_spline_basis(degree)

    y = coef2curve_Spline(basis_function, x_eval, extended_grid, coef, σ)
    @test size(y) == (i, o, b)
    @test !any(isnan.(y))

    recovered_coef = curve2coef(basis_function, x_eval, y, extended_grid, σ)
    @test size(recovered_coef) == size(coef)
    y_reconstructed =
        coef2curve_Spline(basis_function, x_eval, extended_grid, recovered_coef, σ)
    return @test norm(y - y_reconstructed) < Float32(0.1)
end

function test_RBF_basis()
    Random.seed!(42)
    x_eval = rand(Float32, i, b) |> pu
    Random.seed!(42)
    grid = rand(Float32, i, g) |> pu
    coef = rand(Float32, i, o, g) |> pu

    scale = (maximum(grid) - minimum(grid)) / (size(grid, 2) - 1)
    basis_function = RBF_basis(scale)

    y = coef2curve_Spline(basis_function, x_eval, grid, coef, σ)
    @test size(y) == (i, o, b)
    @test !any(isnan.(y))

    recovered_coef = curve2coef(basis_function, x_eval, y, grid, σ)
    @test size(recovered_coef) == size(coef)
    y_reconstructed = coef2curve_Spline(basis_function, x_eval, grid, recovered_coef, σ)
    return @test norm(y - y_reconstructed) < Float32(0.1)
end

function test_RSWAF_basis()
    Random.seed!(42)
    x_eval = rand(Float32, i, b) |> pu
    Random.seed!(42)
    grid = rand(Float32, i, g) |> pu
    coef = rand(Float32, i, o, g) |> pu

    basis_function = RSWAF_basis()

    y = coef2curve_Spline(basis_function, x_eval, grid, coef, σ)
    @test size(y) == (i, o, b)
    @test !any(isnan.(y))

    recovered_coef = curve2coef(basis_function, x_eval, y, grid, σ)
    @test size(recovered_coef) == size(coef)
    y_reconstructed = coef2curve_Spline(basis_function, x_eval, grid, recovered_coef, σ)
    return @test norm(y - y_reconstructed) < Float32(0.1)
end

function test_FFT_basis()
    Random.seed!(42)
    x_eval = rand(Float32, i, b) |> pu
    Random.seed!(42)
    grid = rand(Float32, i, g) |> pu
    coef = rand(Float32, 2, i, o, g) |> pu

    basis_function = FFT_basis()

    y = coef2curve_FFT(basis_function, x_eval, grid, coef, σ)
    @test size(y) == (i, o, b)
    return @test !any(isnan.(y))
end

function test_Cheby_basis()
    Random.seed!(42)
    x_eval = rand(Float32, i, b) |> pu
    Random.seed!(42)
    grid = rand(Float32, i, g) |> pu
    coef = rand(Float32, i, o, degree + 1) |> pu

    basis_function = Cheby_basis(degree)

    y = coef2curve_Spline(basis_function, x_eval, grid, coef, σ)
    @test size(y) == (i, o, b)
    @test !any(isnan.(y))

    recovered_coef = curve2coef(basis_function, x_eval, y, grid, σ)
    @test size(recovered_coef) == size(coef)
    y_reconstructed = coef2curve_Spline(basis_function, x_eval, grid, recovered_coef, σ)
    return @test norm(y - y_reconstructed) < Float32(0.1)
end

@testset "Spline Tests" begin
    test_extend_grid()
    # test_B_spline_basis()
    test_RBF_basis()
    test_RSWAF_basis()
    test_FFT_basis()
    test_Cheby_basis()
end
