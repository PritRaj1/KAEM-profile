module LstSqSolver

export regularize, cholesky_solve

using Lux, LinearAlgebra

function regularize(B_i, y_i, basis; ε = 1.0f-4, init = false)
    J, O, G = basis.I, basis.O, basis.G
    S = init ? size(B_i, 2) : basis.S

    B_perm = reshape(B_i, G, 1, S, 1, J)
    B_perm_transpose = reshape(B_perm, 1, G, S, 1, J)
    A = dropdims(
        sum(
            B_perm .* B_perm_transpose; dims = 3
        ); dims = (3, 4)
    ) # G x G x J

    y_perm = reshape(y_i, 1, 1, S, O, J)
    b = dropdims(
        sum(
            B_perm .* y_perm; dims = 3
        ); dims = (2, 3)
    ) # G x O x J

    eye = (1:G .== (1:G)') |> Lux.f32
    A = A .+ ε .* eye
    return A .* 1.0f0, b .* 1.0f0
end

function _batched_cholesky_solve(A::Array{T, 3}, b::Array{T, 3}) where {T}
    J = size(A, 3)
    coef = similar(b)
    for j in 1:J
        F = cholesky(Symmetric(@view A[:, :, j]))
        coef[:, :, j] = F \ @view b[:, :, j]
    end
    return coef
end

function _batched_cholesky_solve(A, b)
    F = cholesky(A)
    return F \ b
end

function cholesky_solve(A, b)
    return _batched_cholesky_solve(A .* 1.0f0, b .* 1.0f0)
end

end
