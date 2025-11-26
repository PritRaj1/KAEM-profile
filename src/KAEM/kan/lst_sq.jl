module LstSqSolver

export regularize, forward_elimination, backward_substitution, ForwardElim, BackSub

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
        k_mask,
        k_mask_transposed,
        lower_mask,
        upper_mask,
        upper_mask_transposed,
    )
    pivot = sum(A .* k_mask .* k_mask_transposed, dims = (1, 2))
    pivot_row = sum(A .* k_mask; dims = 1)
    pivot_col = sum(A .* k_mask_transposed; dims = 2)

    factors = pivot_col .* lower_mask ./ pivot

    # Rank-1 update
    elimination_mask = lower_mask .* upper_mask_transposed
    A = A .- (factors .* pivot_row) .* elimination_mask

    pivot_b = sum(b .* k_mask; dims = 1)
    b = b .- factors .* pivot_b
    return A, b
end


function forward_elimination(
        A,
        b,
        basis,
    )
    G = basis.G
    k_mask_all = basis.k_mask .* 1.0f0
    k_mask_transposed_all = basis.k_mask_transposed .* 1.0f0
    lower_mask_all = basis.lower_mask .* 1.0f0
    upper_mask_all = basis.upper_mask .* 1.0f0
    upper_mask_transposed_all = basis.upper_mask_transposed .* 1.0f0

    state = (A, b)
    for k in 1:(G - 1)
        A_acc, b_acc = state
        A_acc, b_acc = eliminator(
            k,
            A_acc,
            b_acc,
            @inbounds(selectdim(k_mask_all, 2, k)),
            @inbounds(selectdim(k_mask_transposed_all, 3, k)),
            @inbounds(selectdim(lower_mask_all, 2, k)),
            @inbounds(selectdim(upper_mask_all, 2, k)),
            @inbounds(selectdim(upper_mask_transposed_all, 3, k)),
        )
        state = (A_acc, b_acc)
    end
    return state
end

function backsubber(
        k,
        coef,
        A,
        b,
        k_mask,
        k_mask_transposed,
        upper_mask,
    )
    diag_elem = sum(A .* k_mask .* k_mask_transposed; dims = (1, 2))
    rhs_elem = sum(b .* k_mask; dims = 1)

    upper_row = sum(A .* k_mask; dims = 1)
    upper_coef = PermutedDimsArray(coef .* upper_mask, (2, 1, 3, 4))
    sum_term = sum(upper_row .* upper_coef; dims = 2)

    new_coef_k = (rhs_elem .- sum_term) ./ diag_elem
    coef = coef .+ new_coef_k .* k_mask
    return coef
end


function backward_substitution(
        A,
        b,
        basis,
    )
    k_mask_all = basis.k_mask .* 1.0f0
    k_mask_transposed_all = basis.k_mask_transposed .* 1.0f0
    upper_mask_all = basis.lower_mask .* 1.0f0
    G = basis.G

    coef = zero(b)
    for k in G:-1:1
        coef = backsubber(
            k,
            coef,
            A,
            b,
            @inbounds(selectdim(k_mask_all, 2, k)),
            @inbounds(selectdim(k_mask_transposed_all, 3, k)),
            @inbounds(selectdim(upper_mask_all, 2, k)),
        )
    end

    return coef
end

end
