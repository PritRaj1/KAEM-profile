module GridUpdating

export update_fcn_grid

using Accessors, ComponentArrays, Lux, NNlib, LinearAlgebra, Random

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
    sample_size = size(x, 2)
    coef = ps.coef
    τ = l.τ_trainable ? ps.basis_τ : st.basis_τ

    x_sort = sort(x, dims = 2)
    y =
        l.spline_string == "FFT" ?
        coef2curve_FFT(l.basis_function, x_sort, st.grid, coef, τ) :
        coef2curve_Spline(l.basis_function, x_sort, st.grid, coef, τ)

    # Adaptive grid - concentrate grid points around regions of higher density
    num_interval = size(st.grid, 2) - 2 * l.spline_degree - 1
    ids = reshape([div(sample_size * i, num_interval) + 1 for i in 0:(num_interval - 1)], 1, 1, :)
    mask = ids .== (1:sample_size)'
    grid_adaptive = dropdims(sum(mask .* x_sort; dims = 2); dims = 2)
    grid_adaptive = hcat(grid_adaptive, view(x_sort, :, sample_size))

    # Uniform grid
    h = (view(grid_adaptive, :, num_interval) .- view(grid_adaptive, :, 1)) ./ num_interval # step size
    range = (0:num_interval)' |> pu
    grid_uniform = h .* range .+ view(grid_adaptive, :, 1)

    # Grid is a convex combination of the uniform and adaptive grid
    grid = l.grid_update_ratio .* grid_uniform + (1 - l.grid_update_ratio) .* grid_adaptive
    new_grid = extend_grid(grid; k_extend = l.spline_degree)
    new_coef =
        l.spline_string == "FFT" ? curve2coef(l.basis_function, x_sort, y, new_grid, τ; ε = l.ε_ridge) :
        curve2coef(l.basis_function, x_sort, y, new_grid, τ; ε = l.ε_ridge)

    return new_grid, new_coef
end

end
