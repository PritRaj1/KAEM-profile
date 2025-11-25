module spline_functions

export extend_grid,
    coef2curve_FFT,
    coef2curve_Spline,
    curve2coef,
    B_spline_basis,
    RBF_basis,
    RSWAF_basis,
    FFT_basis,
    Cheby_basis

using ComponentArrays, LinearAlgebra, Lux

using ..Utils

include("lst_sq.jl")
using .LstSqSolver

function extend_grid(
        grid;
        k_extend = 0,
    )
    h = (grid[:, end] - grid[:, 1]) / (size(grid, 2) - 1)

    for i in 1:k_extend
        grid = hcat(grid[:, 1:1] .- h, grid)
        grid = hcat(grid, grid[:, end:end] .+ h)
    end

    return grid
end

struct B_spline_basis <: AbstractBasis
    degree::Int
    I::Int
    O::Int
    G::Int
    S::Int
    k_mask
    lower_mask
    upper_mask
end

function B_spline_basis(degree::Int, I::Int, O::Int, G::Int, S::Int)
    k_mask = Float32.((1:G) .== (1:G)') .* 1.0f0
    lower_mask = Float32.((1:G) .> (1:G)') .* 1.0f0
    upper_mask = Float32.((1:G) .>= (1:G)') .* 1.0f0
    return B_spline_basis(degree, I, O, G, S, k_mask, lower_mask, upper_mask)
end

struct RBF_basis <: AbstractBasis
    I::Int
    O::Int
    G::Int
    S::Int
    k_mask
    lower_mask
    upper_mask
end

function RBF_basis(I::Int, O::Int, G::Int, S::Int)
    k_mask = Float32.((1:G) .== (1:G)') .* 1.0f0
    lower_mask = Float32.((1:G) .> (1:G)') .* 1.0f0
    upper_mask = Float32.((1:G) .>= (1:G)') .* 1.0f0
    return RBF_basis(I, O, G, S, k_mask, lower_mask, upper_mask)
end

struct RSWAF_basis <: AbstractBasis
    I::Int
    O::Int
    G::Int
    S::Int
    k_mask
    lower_mask
    upper_mask
end

function RSWAF_basis(I::Int, O::Int, G::Int, S::Int)
    k_mask = Float32.((1:G) .== (1:G)') .* 1.0f0
    lower_mask = Float32.((1:G) .> (1:G)') .* 1.0f0
    upper_mask = Float32.((1:G) .>= (1:G)') .* 1.0f0
    return RSWAF_basis(I, O, G, S, k_mask, lower_mask, upper_mask)
end

struct Cheby_basis <: AbstractBasis
    degree::Int
    lin::AbstractArray{Float32}
    I::Int
    O::Int
    G::Int
    S::Int
    k_mask
    lower_mask
    upper_mask
end

function Cheby_basis(degree::Int, I::Int, O::Int, S::Int)
    lin = collect(Float32, 0:degree)'
    G = degree + 1
    k_mask = Float32.((1:G) .== (1:G)') .* 1.0f0
    lower_mask = Float32.((1:G) .> (1:G)') .* 1.0f0
    upper_mask = Float32.((1:G) .>= (1:G)') .* 1.0f0
    return Cheby_basis(degree, lin, I, O, G, S, k_mask, lower_mask, upper_mask)
end

function (b::B_spline_basis)(
        x,
        grid,
        σ,
        scale;
        init::Bool = false,
    )
    I, G, S = b.I, b.G - 1, b.S
    x = init ? reshape(x, I, 1, :) : reshape(x, I, 1, S)

    # B0
    grid_1 = @view grid[:, 1:(end - 1)]
    grid_2 = @view grid[:, 2:end]
    B = Float32.((x .>= grid_1) .* (x .< grid_2))

    # Iteratively build up to degree k
    for d in 1:b.degree
        gmax = G - d - 1
        B1 = @view B[:, 1:gmax, :]
        B2 = @view B[:, 2:(gmax + 1), :]
        g1 = @view grid[:, 1:gmax, :]
        g2 = @view grid[:, 2:(gmax + 1), :]
        g3 = @view grid[:, (d + 1):(d + gmax), :]
        g4 = @view grid[:, (d + 2):(d + gmax + 1), :]

        denom1 = g3 .- g1
        denom2 = g4 .- g2

        mask1 = Float32.(denom1 .!= 0)
        mask2 = Float32.(denom2 .!= 0)

        numer1 = x .- g1
        numer2 = g4 .- x

        B = @. ((numer1 / denom1) * B1 * mask1 + (numer2 / denom2) * B2 * mask2)
    end

    return B
end

function (b::RBF_basis)(
        x,
        grid,
        σ,
        scale;
        init::Bool = false,
    )
    I, G, S = b.I, b.G, b.S
    x_3d = init ? reshape(x, I, 1, :) : reshape(x, I, 1, S)
    return @. exp(-((x_3d - grid) * (scale * σ))^2 / 2)
end

function (b::RSWAF_basis)(
        x,
        grid,
        σ,
        scale;
        init::Bool = false,
    )
    I, G, S = b.I, b.G, b.S
    x_3d = init ? reshape(x, I, 1, :) : reshape(x, I, 1, S)
    diff = @. tanh((x_3d - grid) / σ)
    return @. 1.0f0 - diff^2
end

function (b::Cheby_basis)(
        x,
        grid,
        σ,
        scale;
        init::Bool = false,
    )
    I, S = b.I, b.S
    x = init ? reshape(x, I, 1, :) : reshape(x, I, 1, S)
    x = @. acos(tanh(x) / σ)
    return @. cos(x * b.lin)
end

function coef2curve_Spline(
        b,
        x_eval,
        grid,
        coef,
        σ,
        scale;
        init::Bool = false,
    )
    I, O, G, S = b.I, b.O, b.G, b.S
    spl = b(x_eval, grid, σ, scale)
    spl_4d = reshape(spl, I, 1, S, G)
    coef_4d = reshape(coef, I, O, 1, G)

    return dropdims(
        sum(
            spl_4d .* coef_4d; dims = 4
        ); dims = 4
    )
end

function curve2coef(
        b,
        x,
        y,
        grid,
        σ,
        scale;
        init = false,
        ε = 1.0f-4
    )
    J, O, G = b.I, b.O, b.G
    S = init ? size(x, 2) : b.S
    B = b(x, grid, σ, scale; init = init)
    B = permutedims(B, (2, 3, 1))
    y = permutedims(y, (3, 2, 1))

    A, b_vec = regularize(B, y, J, O, G, S; ε = ε)
    A, b_vec = forward_elimination(A, b_vec, G, b.k_mask, b.lower_mask, b.upper_mask)
    coef = dropdims(backward_substitution(A, b_vec, G, b.k_mask, b.lower_mask); dims = 2)
    return permutedims(coef, (3, 2, 1))
end

## FFT basis functions ###
struct FFT_basis <: AbstractBasis
    I::Int
    O::Int
    G::Int
    S::Int
end

function (b::FFT_basis)(
        x,
        grid,
        σ
    )
    I, G, S = b.I, b.G, b.S

    x_3d = reshape(x, I, 1, S)
    grid_3d = reshape(grid, I, G, 1)
    freq = @. x_3d * grid_3d * Float32(2π) * σ
    return cos.(freq), sin.(freq)
end

function coef2curve_FFT(
        b,
        x_eval,
        grid,
        coef,
        σ,
    )
    I, O, G, S = b.I, b.O, b.G, b.S

    even, odd = b(x_eval, grid, σ)
    even = reshape(permutedims(even, (1, 3, 2)), I, 1, S, G)
    odd = reshape(permutedims(odd, (1, 3, 2)), I, 1, S, G)
    even_coef = reshape(coef[1, :, :, :], I, O, 1, G)
    odd_coef = reshape(coef[2, :, :, :], I, O, 1, G)

    y_even = sum(even .* even_coef; dims = 4)
    y_odd = sum(odd .* odd_coef; dims = 4)
    return dropdims(y_even + y_odd; dims = 4)
end

end
