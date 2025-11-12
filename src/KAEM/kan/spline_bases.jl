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

using ComponentArrays, LinearAlgebra, NNlib

using ..Utils

function extend_grid(
        grid::AbstractArray{T, 2};
        k_extend::Int = 0,
    )::AbstractArray{T, 2} where {T <: Float32}
    h = (grid[:, end] - grid[:, 1]) / (size(grid, 2) - 1)

    for i in 1:k_extend
        grid = hcat(grid[:, 1:1] .- h, grid)
        grid = hcat(grid, grid[:, end:end] .+ h)
    end

    return grid
end

struct B_spline_basis <: AbstractBasis
    degree::Int
end

struct RBF_basis <: AbstractBasis
    scale::Float32
end

struct RSWAF_basis <: AbstractBasis end

struct Cheby_basis <: AbstractBasis
    degree::Int
    lin::AbstractArray{Float32}
end

function Cheby_basis(degree::Int)
    lin = collect(Float32, 0:degree)'
    return Cheby_basis(degree, lin)
end

function (b::B_spline_basis)(
        x,
        grid,
        σ
    )
    I, S, G = size(x)..., size(grid, 2)
    x = reshape(x, I, 1, S)

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
    )
    I, S, G = size(x)..., size(grid, 2)
    x_3d = reshape(x, I, 1, S)
    grid_3d = reshape(grid, I, G, 1)
    return @. exp(-((x_3d - grid_3d) * (b.scale * σ))^2 / 2)
end

function (b::RSWAF_basis)(
        x,
        grid,
        σ,
    )
    I, S, G = size(x)..., size(grid, 2)
    diff = NNlib.tanh_fast((reshape(x, I, 1, S) .- grid) ./ σ)
    return @. 1.0f0 - diff^2
end

function (b::Cheby_basis)(
        x,
        grid,
        σ,
    )
    I, S = size(x)
    z = acos.(NNlib.tanh_fast(x) ./ σ)
    return cos.(reshape(z, I, 1, S) .* b.lin)
end

function coef2curve_Spline(
        b,
        x_eval,
        grid,
        coef,
        σ,
    )
    spl = b(x_eval, grid, σ)
    I, G, S, O = size(spl)..., size(coef, 2)
    return dropdims(
        sum(
            reshape(spl, I, 1, S, G) .* reshape(coef, I, O, 1, G); dims = 4
        ); dims = 4
    )
end

function curve2coef(
        b,
        x,
        y,
        grid,
        σ;
        init = false
    )
    J, S, O = size(x)..., size(y, 2)

    B = b(x, grid, σ)
    G = size(B, 2)

    B = reshape(B, S, J * G)
    y = reshape(y, S, J * O)
    eye = Array{Float32}(I, J, J)
    coef = reshape(B \ y, J, J, O, G)

    return dropdims(sum(coef .* eye, dims = 1); dims = 1)
end

## FFT basis functions ###
struct FFT_basis <: AbstractBasis end

function (b::FFT_basis)(
        x,
        grid,
        σ,
    )
    I, S, G = size(x)..., size(grid, 2)

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
    even, odd = b(x_eval, grid, σ)
    even_coef = @view coef[1, :, :, :]
    odd_coef = @view coef[2, :, :, :]
    I, G, S, O = size(even)..., size(odd_coef, 2)

    y_even = sum(reshape(even, I, 1, S, G) .* reshape(even_coef, I, O, 1, G); dims = 4)
    y_odd = sum(reshape(odd, I, 1, S, G) .* reshape(odd_coef, I, O, 1, G); dims = 4)
    return dropdims(y_even + y_odd; dims = 4)
end

end
