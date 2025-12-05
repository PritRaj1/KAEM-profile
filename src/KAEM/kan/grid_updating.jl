module GridUpdating

export update_fcn_grid

using Accessors, ComponentArrays, Lux, NNlib, LinearAlgebra

using ..Utils
using ..UnivariateFunctions
using ..UnivariateFunctions.spline_functions

function update_fcn_grid(
        l,
        ps,
        st,
        x,
    )
    """
    Adapt the function's grid to the distribution of the input data.

    Args:
        l: The univariate function layer.
        ps: The parameters of the layer.
        st: The state of the KAN layer.
        x_p: The input of size (i, num_samples).

    Returns:
        new_grid: The updated grid.
        new_coef: The updated spline coefficients.
    """
    coef = ps.coef
    τ = l.τ_trainable ? ps.basis_τ : st.basis_τ
    I, S, G = l.basis_function.I, l.basis_function.S, l.basis_function.G

    x_sort = sort(x, dims = 2)
    y =
        l.spline_string == "FFT" ?
        coef2curve_FFT(l.basis_function, x_sort, st.grid, coef, τ) :
        coef2curve_Spline(l.basis_function, x_sort, st.grid, coef, τ, st.scale)

    # Adaptive grid - concentrate grid points around regions of higher density
    num_interval = G - 2 * l.spline_degree - 1
    ids = [div(S * i, num_interval) + 1 for i in 0:(num_interval - 1)]
    mask = PermutedDimsArray(view(ids .== (1:S)', :, :, :), (3, 2, 1))
    grid_adaptive = dropdims(sum(mask .* x_sort; dims = 2); dims = 2)
    grid_adaptive = hcat(grid_adaptive, x_sort[:, S:S])

    # Uniform grid
    h = (grid_adaptive[:, num_interval:num_interval] .- grid_adaptive[:, 1:1] .* 1.0f0) ./ num_interval # step size
    range = (0:num_interval)' |> pu
    grid_uniform = h .* range .+ grid_adaptive[:, 1:1] .* 1.0f0

    # Grid is a convex combination of the uniform and adaptive grid
    grid = l.grid_update_ratio .* grid_uniform + (1 - l.grid_update_ratio) .* grid_adaptive
    new_grid = extend_grid(grid; k_extend = l.spline_degree)
    new_coef = curve2coef(
        l.basis_function,
        x_sort,
        y,
        new_grid,
        τ,
        st.scale;
        ε = l.ε_ridge
    )

    return new_grid, new_coef
end

end
