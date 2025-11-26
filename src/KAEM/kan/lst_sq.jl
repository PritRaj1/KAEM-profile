module LstSqSolver

export regularize, forward_elimination, backward_substitution, ForwardElim, BackSub

using Lux
using Reactant: @trace

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
        lower_mask,
        upper_mask,
    )
    pivot = sum(A .* k_mask .* k_mask', dims = (1, 2))
    pivot_row = sum(A .* k_mask; dims = 1)
    pivot_col = sum(A .* k_mask'; dims = 2)

    factors = pivot_col .* lower_mask ./ pivot

    # Rank-1 update
    elimination_mask = lower_mask .* upper_mask'
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
    k_mask_all = Lux.f32(basis.k_mask) .* 1.0f0
    lower_mask_all = Lux.f32(basis.lower_mask) .* 1.0f0
    upper_mask_all = Lux.f32(basis.upper_mask) .* 1.0f0

    state = (1, A, b)
    @trace while first(state) < G
        k, A_acc, b_acc = state
        k_mask = k_mask_all[:, k:k]
        lower_mask = lower_mask_all[:, k:k]
        upper_mask = upper_mask_all[:, k:k]
        A_acc, b_acc = eliminator(
            k,
            A_acc,
            b_acc,
            k_mask,
            lower_mask,
            upper_mask,
        )
        state = (k + 1, A_acc, b_acc)
    end

    _, A, b = state
    return A, b
end

function backsubber(
        k,
        coef,
        A,
        b,
        k_mask,
        upper_mask,
    )
    diag_elem = sum(A .* k_mask .* k_mask'; dims = (1, 2))
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
    G = basis.G
    k_mask_all = Lux.f32(basis.k_mask) .* 1.0f0
    upper_mask_all = Lux.f32(basis.lower_mask) .* 1.0f0

    state = (G, zero(b))
    @trace while first(state) > 0
        k, coef_acc = state
        k_mask = k_mask_all[:, k:k]
        upper_mask = upper_mask_all[:, k:k]
        coef_acc = backsubber(
            k,
            coef_acc,
            A,
            b,
            k_mask,
            upper_mask,
        )
        state = (k - 1, coef_acc)
    end

    return last(state)
end

end
