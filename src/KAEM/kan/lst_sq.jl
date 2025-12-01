module LstSqSolver

export regularize, eliminator, backsubber

using Lux

function regularize(B_i, y_i, basis; ε = 1.0f-4, init = false)
    J, O, G = basis.I, basis.O, basis.G
    S = init ? size(B_i, 2) : basis.S

    B_perm = reshape(B_i, G, 1, S, 1, J)
    B_perm_transpose = reshape(B_perm, 1, G, S, 1, J)
    A = dropdims(
        sum(
            B_perm .* B_perm_transpose; dims = 3
        ); dims = 3
    ) # G x G x 1 x J

    y_perm = reshape(y_i, 1, 1, S, O, J)
    b = dropdims(
        sum(
            B_perm .* y_perm; dims = 3
        ); dims = 3
    ) # G x 1 x O x J

    eye = 1:G .== (1:G)' |> Lux.f32
    A = @. A + ε * eye
    return A .* 1.0f0, b .* 1.0f0
end

function eliminator(
        k,
        A,
        b,
        lower_mask_all,
        upper_mask_all,
    )
    lower_mask = lower_mask_all[:, k]
    upper_mask = upper_mask_all[:, k]

    pivot = A[k:k, k:k, :, :]
    pivot_row = A[k:k, :, :, :]
    pivot_col = A[:, k:k, :, :]

    factors = pivot_col .* lower_mask ./ pivot

    # Rank-1 update
    elimination_mask = lower_mask * upper_mask'
    A = A .- (factors .* pivot_row) .* elimination_mask

    pivot_b = b[k:k, :, :, :]
    b = b .- factors .* pivot_b
    return A, b
end

function backsubber(
        k,
        coef,
        A,
        b,
        k_mask_all,
        upper_mask_all,
        J, O, G
    )
    k_mask = k_mask_all[:, k]
    upper_mask = upper_mask_all[:, k]

    diag_elem = A[k:k, k:k, :, :]
    rhs_elem = b[k:k, :, :, :]

    upper_row = A[k:k, :, :, :]
    upper_coef = PermutedDimsArray(coef .* upper_mask, (2, 1, 3, 4))
    sum_term = sum(upper_row .* upper_coef; dims = 2)

    new_coef_k = (rhs_elem .- sum_term) ./ diag_elem
    coef = coef .+ new_coef_k .* k_mask
    return coef
end

end
