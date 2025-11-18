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

using ComponentArrays, LinearAlgebra

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
    I::Int
    O::Int
    G::Int
end

struct RBF_basis <: AbstractBasis
    scale::Float32
    I::Int
    O::Int
    G::Int
end

struct RSWAF_basis <: AbstractBasis
    I::Int
    O::Int
    G::Int
end

struct Cheby_basis <: AbstractBasis
    degree::Int
    lin::AbstractArray{Float32}
    I::Int
    O::Int
    G::Int
end

function Cheby_basis(degree::Int, I::Int, O::Int)
    lin = collect(Float32, 0:degree)'
    return Cheby_basis(degree, lin, I, O, degree + 1)
end

function (b::B_spline_basis)(
        x,
        grid,
        σ
    )
    I, G = b.I, b.G - 1
    x = reshape(x, I, 1, :)

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
    I, G = b.I, b.G
    x_3d = reshape(x, I, 1, :)
    return @. exp(-((x_3d - grid) * (b.scale * σ))^2 / 2)
end

function (b::RSWAF_basis)(
        x,
        grid,
        σ,
    )
    I, G = b.I, b.G
    x_3d = reshape(x, I, 1, :)
    diff = @. tanh((x_3d - grid) / σ)
    return @. 1.0f0 - diff^2
end

function (b::Cheby_basis)(
        x,
        grid,
        σ,
    )
    I = b.I
    x = reshape(x, I, 1, :)
    x = @. acos(tanh(x) / σ)
    return @. cos(x * b.lin)
end

function coef2curve_Spline(
        b,
        x_eval,
        grid,
        coef,
        σ,
    )
    I, O, G = b.I, b.O, b.G
    spl = b(x_eval, grid, σ)
    spl_4d = reshape(spl, I, 1, :, G)
    coef_4d = reshape(coef, I, O, 1, G)

    return dropdims(
        sum(
            spl_4d .* coef_4d; dims = 4
        ); dims = 4
    )
end

function ridge_regression(B, y, i, o, G; ε = 1.0f-4)
    """Here, '\' needs rows = measurements/samples."""
    B_i = view(B, :, :, i)
    y_i = view(y, :, o, i)

    λ = ε .* Array{Float32}(I, G, G)
    BtB = B_i' * B_i .+ λ
    Bty = B_i' * y_i
    return reshape(BtB \ Bty, 1, 1, G)
end

function curve2coef(
        b,
        x,
        y,
        grid,
        σ;
        init = false,
        ε = 1.0f-4
    )
    J, O, G = b.I, b.O, b.G
    B = b(x, grid, σ)

    B_perm = permutedims(B, (3, 2, 1)) # S, G, I
    y_perm = permutedims(y, (3, 2, 1)) # S, O, I

    # Least squares for each input and output.
    coef = reduce(
        vcat,
        map(
            i -> reduce(
                hcat,
                map(
                    o -> ridge_regression(
                        B_perm,
                        y_perm,
                        i, o, G;
                        ε = ε
                    ), 1:O
                )
            ), 1:J
        )
    )

    return coef
end

## FFT basis functions ###
struct FFT_basis <: AbstractBasis
    I::Int
    O::Int
    G::Int
end

function (b::FFT_basis)(
        x,
        grid,
        σ,
    )
    I, G = b.I, b.G

    x_3d = reshape(x, I, 1, :)
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
    I, O, G = b.I, b.O, b.G

    even, odd = b(x_eval, grid, σ)
    even = reshape(even, I, 1, :, G)
    odd = reshape(odd, I, 1, :, G)
    even_coef = reshape(coef[1, :, :, :], I, O, 1, G)
    odd_coef = reshape(coef[2, :, :, :], I, O, 1, G)

    y_even = sum(even .* even_coef; dims = 4)
    y_odd = sum(odd .* odd_coef; dims = 4)
    return dropdims(y_even + y_odd; dims = 4)
end

end
