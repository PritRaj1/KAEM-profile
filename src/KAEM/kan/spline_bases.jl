module spline_functions

export extend_grid,
    coef2curve_FFT,
    coef2curve_Spline,
    curve2coef,
    B_spline_basis,
    RBF_basis,
    RSWAF_basis,
    FFT_basis,
    Cheby_basis,
    SplineMUL

using CUDA, Lux, ComponentArrays
using LinearAlgebra, NNlib

using ..Utils

function extend_grid(
    grid::AbstractArray{T,2};
    k_extend::Int = 0,
)::AbstractArray{T,2} where {T<:half_quant}
    h = (grid[:, end] - grid[:, 1]) / (size(grid, 2) - 1)

    for i = 1:k_extend
        grid = hcat(grid[:, 1:1] .- h, grid)
        grid = hcat(grid, grid[:, end:end] .+ h)
    end

    return grid
end

function SplineMUL(
    l::Lux.AbstractLuxLayer,
    ps::ComponentArray{T},
    x::AbstractArray{T,2},
    y::AbstractArray{T,3},
)::AbstractArray{T,3} where {T<:half_quant}
    x_act = l.base_activation(x)
    w_base, w_sp = ps.w_base, ps.w_sp
    I, S, O = size(x_act)..., size(w_base, 2)
    return reshape(w_base, I, O, 1) .* reshape(x_act, I, 1, S) .+
           reshape(w_sp, I, O, 1) .* y
end

struct B_spline_basis <: AbstractBasis
    degree::Int
end

struct RBF_basis <: AbstractBasis
    scale::half_quant
end

struct RSWAF_basis <: AbstractBasis end

struct Cheby_basis <: AbstractBasis
    degree::Int
    lin::AbstractArray{half_quant}
end

function Cheby_basis(degree::Int)
    lin = collect(half_quant, 0:degree)' |> pu
    return Cheby_basis(degree, lin)
end

function (b::B_spline_basis)(
    x::AbstractArray{T,2},
    grid::AbstractArray{T,2},
    σ::AbstractArray{T,1};
)::AbstractArray{T,3} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    x = reshape(x, I, 1, S)

    # B0
    grid_1 = @view grid[:, 1:(end-1)]
    grid_2 = @view grid[:, 2:end]
    B = T.((x .>= grid_1) .* (x .< grid_2))

    # Iteratively build up to degree k
    for d = 1:b.degree
        gmax = G - d - 1
        B1 = @view B[:, 1:gmax, :]
        B2 = @view B[:, 2:(gmax+1), :]
        g1 = @view grid[:, 1:gmax, :]
        g2 = @view grid[:, 2:(gmax+1), :]
        g3 = @view grid[:, (d+1):(d+gmax), :]
        g4 = @view grid[:, (d+2):(d+gmax+1), :]

        denom1 = g3 .- g1
        denom2 = g4 .- g2

        mask1 = T.(denom1 .!= 0)
        mask2 = T.(denom2 .!= 0)

        numer1 = x .- g1
        numer2 = g4 .- x

        B = @. ((numer1 / denom1) * B1 * mask1 + (numer2 / denom2) * B2 * mask2)
    end

    return B
end

function (b::RBF_basis)(
    x::AbstractArray{T,2},
    grid::AbstractArray{T,2},
    σ::AbstractArray{T,1},
)::AbstractArray{T,3} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    x_3d = reshape(x, I, 1, S)
    grid_3d = reshape(grid, I, G, 1)
    return @. exp(-((x_3d - grid_3d) * (b.scale * σ))^2 / 2)
end

function (b::RSWAF_basis)(
    x::AbstractArray{T,2},
    grid::AbstractArray{T,2},
    σ::AbstractArray{T,1},
)::AbstractArray{T,3} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    diff = NNlib.tanh_fast((reshape(x, I, 1, S) .- grid) ./ σ)
    return @. one(T) - diff^2
end

function (b::Cheby_basis)(
    x::AbstractArray{T,2},
    grid::AbstractArray{T,2},
    σ::AbstractArray{T,1},
)::AbstractArray{T,3} where {T<:half_quant}
    I, S = size(x)
    z = acos.(NNlib.tanh_fast(x) ./ σ)
    return cos.(reshape(z, I, 1, S) .* b.lin)
end

function coef2curve_Spline(
    b::AbstractBasis,
    x_eval::AbstractArray{T,2},
    grid::AbstractArray{T,2},
    coef::AbstractArray{T,3},
    σ::AbstractArray{T,1},
)::AbstractArray{T,3} where {T<:half_quant}
    spl = b(x_eval, grid, σ)
    I, G, S, O = size(spl)..., size(coef, 2)
    coef_perm = permutedims(coef, (2, 3, 1)) # [O, G, I]
    spl_perm = permutedims(spl, (2, 3, 1)) # [G, S, I]
    y_perm = NNlib.batched_mul(coef_perm, spl_perm)
    return permutedims(y_perm, (3, 1, 2)) # [I, O, S]
end

function curve2coef(
    b::AbstractBasis,
    x::AbstractArray{T,2},
    y::AbstractArray{T,3},
    grid::AbstractArray{T,2},
    σ::AbstractArray{T,1},
)::AbstractArray{T,3} where {T<:half_quant}
    J, S, O = size(x)..., size(y, 2)

    B = b(x, grid, σ) .|> full_quant
    y = y .|> full_quant
    G = size(B, 2)

    B = permutedims(B, [1, 3, 2]) # in_dim x b_size x n_grid

    coef = Array{full_quant}(undef, J, O, G) |> pu
    for i = 1:J
        for o = 1:O
            coef[i, o, :] .= B[i, :, :] \ y[i, o, :]
        end
    end

    return T.(coef)
end

## FFT basis functions ###
struct FFT_basis <: AbstractBasis end

function (b::FFT_basis)(
    x::AbstractArray{T,2},
    grid::AbstractArray{T,2},
    σ::AbstractArray{T,1},
)::Tuple{AbstractArray{T,3},AbstractArray{T,3}} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)

    x_3d = reshape(x, I, 1, S)
    grid_3d = reshape(grid, I, G, 1)
    freq = @. x_3d * grid_3d * T(2π) * σ
    return cos.(freq), sin.(freq)
end

function coef2curve_FFT(
    b::AbstractBasis,
    x_eval::AbstractArray{T,2},
    grid::AbstractArray{T,2},
    coef::AbstractArray{T,4},
    σ::AbstractArray{T,1},
)::AbstractArray{T,3} where {T<:half_quant}
    even, odd = b(x_eval, grid, σ)
    even_coef = @view coef[1, :, :, :]
    odd_coef = @view coef[2, :, :, :]

    even_coef_perm = permutedims(even_coef, (2, 3, 1))
    odd_coef_perm = permutedims(odd_coef, (2, 3, 1))
    even_perm = permutedims(even, (2, 3, 1))
    odd_perm = permutedims(odd, (2, 3, 1))

    y_even = NNlib.batched_mul(even_coef_perm, even_perm)  # [O, S, I]
    y_odd = NNlib.batched_mul(odd_coef_perm, odd_perm) # [O, S, I]
    return permutedims(y_even .+ y_odd, (3, 1, 2))
end

end
