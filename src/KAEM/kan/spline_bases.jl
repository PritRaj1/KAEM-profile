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

function lst_sq(B_i, y_i, G; ε = 1.0f-4)
    """Reactant-compatible, gaussian elimination"""
    A = B_i * B_i'
    b = B_i * y_i

    # Forward elimination using views
    for k in 1:(G - 1)
        pivot = view(A, k, k)
        pivot = ifelse.(pivot .== 0, ε, pivot)
        pivot_row = view(A, k, k:G)
        pivot_col = view(A, (k + 1):G, k)

        sub_A = view(A, (k + 1):G, k:G)
        sub_b = view(b, (k + 1):G)  # Actually use this!

        factors = pivot_col ./ pivot

        # Rank-1 update
        sub_A .-= factors * transpose(pivot_row)

        # Update RHS using sub_b
        pivot_b = view(b, k)
        sub_b .-= factors .* pivot_b
    end

    # Back substitution using ifelse
    x = zero(b)
    for k in G:-1:1
        diag_elem = view(A, k, k)
        diag_elem = ifelse.(diag_elem .== 0, ε, diag_elem)
        rhs_elem = view(b, k)

        # Use ifelse instead of if-else
        sum_term = ifelse(
            k == G,
            zero(rhs_elem),  # No upper triangular part
            sum(view(A, k, (k + 1):G) .* view(x, (k + 1):G))
        )

        view(x, k) .= (rhs_elem .- sum_term) ./ diag_elem
    end

    return x
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

    # Least squares for each input and output.
    coef = similar(B, J, O, G)
    for i in 1:J
        for o in 1:O
            coef[i, o, :] = lst_sq(
                view(B, i, :, :),
                view(y, i, o, :),
                G; ε = ε
            )
        end
    end

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
