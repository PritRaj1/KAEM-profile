module LstSqSolver

export regularize, forward_elimination, backward_substitution

using Lux

function regularize(B_i, y_i, J, O, G, S; ε = 1.0f-4)
    B_perm = reshape(B_i, G, 1, S, 1, J)
    B_perm_transpose = reshape(B_perm, 1, G, S, 1, J)
    A = dropdims(sum(B_perm .* B_perm_transpose; dims = 3); dims = 3) # G x G x 1 x J

    y_perm = reshape(y_i, 1, 1, S, O, J)
    b = dropdims(sum(B_perm .* y_perm; dims = 3); dims = 3) # G x 1 x O x J

    eye = 1:G .== (1:G)' |> Lux.f32
    A = @. A + ε * eye
    return A, b
end

function forward_elimination(A, b, J, G; ε = 1.0f-4)
    for k in 1:(G - 1)
        k_mask = (1:G) .== k |> Lux.f32
        lower_mask = (1:G) .> k |> Lux.f32
        upper_mask = (1:G) .>= k |> Lux.f32

        pivot = sum(A .* k_mask .* k_mask', dims = (1, 2))
        pivot_row = sum(A .* k_mask; dims = 1)
        pivot_col = sum(A .* k_mask'; dims = 2)

        factors = pivot_col .* lower_mask ./ pivot

        # Rank-1 update
        elimination_mask = lower_mask .* upper_mask'
        A = A .- (factors .* pivot_row) .* elimination_mask

        pivot_b = sum(b .* k_mask; dims = 1)
        b = b .- factors .* pivot_b
    end
    return A, b
end

function backward_substitution(A, b, J, G; ε = 1.0f-4)
    coef = zero(b)

    for k in G:-1:1
        k_mask = (1:G) .== k |> Lux.f32
        upper_mask = (1:G) .> k |> Lux.f32

        diag_elem = sum(A .* k_mask .* k_mask'; dims = (1, 2))
        rhs_elem = sum(b .* k_mask; dims = 1)

        upper_row = sum(A .* k_mask; dims = 1)
        upper_coef = permutedims(coef .* upper_mask, (2, 1, 3, 4))
        sum_term = sum(upper_row .* upper_coef; dims = 2)

        new_coef_k = (rhs_elem .- sum_term) ./ diag_elem
        coef = coef .+ new_coef_k .* k_mask
    end

    return coef
end

end
