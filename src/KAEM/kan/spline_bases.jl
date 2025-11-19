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
using Reactant: @allowscalar

using ..Utils

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
end

struct RBF_basis <: AbstractBasis
    scale
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

function regularize(B_i, y_i, J, O, G; ε = 1.0f-4)
    B_perm = reshape(B_i, G, 1, :, 1, J)
    B_perm_transpose = reshape(B_perm, 1, G, :, 1, J)
    A = dropdims(sum(B_perm .* B_perm_transpose; dims = 3); dims = (3, 4)) # G x G x 1 x J

    y_perm = reshape(y_i, 1, 1, :, O, J)
    b = dropdims(sum(B_perm .* y_perm; dims = 3); dims = (2, 3)) # G x O x J

    eye = 1:G .== (1:G)' |> Lux.f32
    @. A += ε * eye
    return A, b
end

function forward_elimination(A, b, J, G; ε = 1.0f-4)
    for k in 1:(G - 1)
        pivot = view(A, k:k, k:k, :)
        pivot_row = view(A, k:k, k:G, :)
        pivot_col = view(A, (k + 1):G, k:k, :)

        factors = pivot_col ./ pivot
        @allowscalar A[(k + 1):G, k:G, :] = view(A, (k + 1):G, k:G, :) .- factors .* pivot_row
        @allowscalar b[(k + 1):G, :, :] = view(b, (k + 1):G, :, :) .- factors .* view(b, k:k, :, :)
    end
    return A, b
end

function backward_substitution(A, b, J, G; ε = 1.0f-4)
    coef = zero(b)
    A_expanded = reshape(A, G, G, 1, J)
    diag_elem = view(A_expanded, G, G, :, :)
    rhs_elem = view(b, G, :, :)
    coef[G, :, :] = rhs_elem ./ diag_elem

    for k in (G - 1):-1:1
        diag_elem = view(A_expanded, k, k, :, :)
        rhs_elem = view(b, k, :, :)
        sum_term = dropdims(
            sum(
                view(A_expanded, k, (k + 1):G, :, :) .*
                    view(coef, (k + 1):G, :, :);
                dims = 1
            )
            ; dims = 1
        )
        @allowscalar coef[k, :, :] = (rhs_elem .- sum_term) ./ diag_elem
    end

    return coef
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
    B = permutedims(B, (2, 3, 1))
    y = permutedims(y, (3, 2, 1))

    A, b = regularize(B, y, J, O, G; ε = ε)
    A, b = forward_elimination(A, b, J, G; ε = ε)
    coef = backward_substitution(A, b, J, G; ε = ε)
    return permutedims(coef, (3, 2, 1))
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
