module ModelGridUpdating

export update_model_grid

using Accessors, ComponentArrays, Lux, NNlib, LinearAlgebra, Random

using ..Utils
using ..KAEM_model
using ..KAEM_model.UnivariateFunctions
using ..KAEM_model.EBM_Model

include("kan/grid_updating.jl")
using .GridUpdating

function update_model_grid(
        model,
        x,
        ps,
        st_kan,
        st_lux,
        train_idx,
        swap_replica_idxs,
        grid_sample_idxs,
        rng,
    )
    """
    Update the grid of the likelihood model using samples from the prior.

    Args:
        model: The model.
        x: Data samples.
        ps: The parameters of the model.
        st_kan: The states of the KAN model.
        st_lux: The states of the Lux model.
        temps: The temperatures for thermodynamic models.
        rng: The random number generator.

    Returns:
        The updated model.
        The updated params.
        The updated KAN states.
        The updated Lux states. 
    """

    z = nothing
    if model.update_prior_grid

        if model.N_t > 1
            temps = collect(Float32, [(k / model.N_t)^model.p[train_idx] for k in 1:model.N_t])
            z = first(
                model.posterior_sampler(
                    ps,
                    st_kan,
                    st_lux,
                    x;
                    temps = temps,
                    rng = rng,
                    swap_replica_idxs = swap_replica_idxs
                ),
            )[
                :,
                :,
                :,
                end,
            ]
        elseif model.prior.bool_config.ula || model.MALA
            z = first(model.posterior_sampler(ps, st_kan, st_lux, x; rng = rng))[
                :,
                :,
                :,
                1,
            ]
        else
            z = first(
                model.sample_prior(
                    model,
                    ps,
                    st_kan,
                    st_lux,
                    rng,
                )
            )
            # z = first(model.posterior_sampler(ps, st_kan, st_lux, x; rng = rng))[
            #     :,
            #     :,
            #     :,
            #     1,
            # ] # For domain updating: use ULA to explore beyond prior init domain.
        end

        # Must update domain for inverse transform sampling
        if (model.MALA || model.N_t > 1 || model.prior.bool_config.ula)
            min_z, max_z = minimum(z), maximum(z)
            st_kan.ebm[:a].min = [min_z * 0.9f0]
            st_kan.ebm[:a].max = [max_z * 1.1f0]
        end

        if !(
                model.prior.fcns_qp[1].spline_string == "FFT" ||
                    model.prior.fcns_qp[1].spline_string == "Cheby"
            )
            Q, P, B = model.prior.q_size, model.prior.p_size, model.batch_size

            # Randomly sample components if univariate to reduce computation
            if !isnothing(grid_sample_idxs)
                mask = Lux.f32(1:Q .== grid_sample_idxs') .* 1.0f0
                z = dropdims(sum(z .* reshape(mask, Q, 1, B); dims = 1); dims = 1)
            end
            z = !isnothing(grid_sample_idxs) ? z : reshape(z, Q, B)

            mid_size = model.prior.bool_config.mixture_model ? P : Q
            outer_dim = model.prior.bool_config.mixture_model ? Q * B : P * B

            for i in 1:model.prior.depth
                if model.prior.bool_config.layernorm && i != 1
                    z, st_ebm = Lux.apply(
                        model.prior.layernorms[i - 1],
                        z,
                        ps.ebm.layernorm[symbol_map[i]],
                        st_lux.ebm[symbol_map[i]],
                    )
                    @reset st_lux.ebm[symbol_map[i]] = st_ebm
                end

                new_grid, new_coef = update_fcn_grid(
                    model.prior.fcns_qp[i],
                    ps.ebm.fcn[symbol_map[i]],
                    st_kan.ebm[symbol_map[i]],
                    z,
                )
                ps.ebm.fcn[symbol_map[i]].coef = new_coef
                st_kan.ebm[symbol_map[i]].grid = new_grid

                if model.prior.fcns_qp[i].spline_string == "RBF"
                    scale = (maximum(new_grid) - minimum(new_grid)) /
                        (size(new_grid, 2) - 1) |> Lux.f32

                    st_kan.ebm[symbol_map[i]].scale = [scale]
                end

                z = Lux.apply(
                    model.prior.fcns_qp[i],
                    z,
                    ps.ebm.fcn[symbol_map[i]],
                    st_kan.ebm[symbol_map[i]],
                )
                z =
                    (i == 1 && !model.prior.bool_config.ula) ? reshape(z, mid_size, outer_dim) :
                    dropdims(sum(z, dims = 1); dims = 1)
            end
        end
    end

    new_nodes, new_weights = get_gausslegendre(model.prior, st_kan.ebm)
    st_kan.quad.nodes = new_nodes
    st_kan.quad.weights = new_weights

    # Only update if KAN-type generator requires
    (!model.update_llhood_grid || model.lkhood.CNN || model.lkhood.SEQ) &&
        return ps, st_kan, st_lux

    if model.N_t > 1
        temps = collect(Float32, [(k / model.N_t)^model.p[train_idx] for k in 1:model.N_t])
        z = first(
            model.posterior_sampler(
                ps,
                st_kan,
                st_lux,
                x;
                temps = temps,
                rng = rng,
                swap_replica_idxs = swap_replica_idxs
            ),
        )[
            :,
            :,
            :,
            end,
        ]
    elseif model.prior.bool_config.ula || model.MALA
        z = first(model.posterior_sampler(ps, st_kan, st_lux, x; rng = rng))[
            :,
            :,
            :,
            1,
        ]
    else
        z = first(
            model.sample_prior(model, ps, st_kan, st_lux, rng)
        )
        # z = first(model.posterior_sampler(ps, st_kan, st_lux, x; rng = rng))[
        #     :,
        #     :,
        #     :,
        #     1,
        # ] # For domain updating: use ULA to explore beyond prior init domain.
    end

    z = dropdims(sum(z; dims = 2); dims = 2)

    for i in 1:model.lkhood.generator.depth
        if model.lkhood.generator.bool_config.layernorm
            z, st_gen = Lux.apply(
                model.lkhood.generator.layernorms[i],
                z,
                ps.gen.layernorm[symbol_map[i]],
                st_lux.gen[symbol_map[i]],
            )
            @reset st_lux.gen[symbol_map[i]] = st_gen
        end

        if !(
                model.lkhood.generator.Φ_fcns[i].spline_string == "FFT" ||
                    model.lkhood.generator.Φ_fcns[i].spline_string == "Cheby"
            )
            new_grid, new_coef = update_fcn_grid(
                model.lkhood.generator.Φ_fcns[i],
                ps.gen.fcn[symbol_map[i]],
                st_kan.gen[symbol_map[i]],
                z,
            )
            ps.gen.fcn[symbol_map[i]].coef = new_coef
            st_kan.gen[symbol_map[i]].grid = new_grid

            if model.lkhood.generator.Φ_fcns[i].spline_string == "RBF"
                scale = (maximum(new_grid) - minimum(new_grid)) /
                    (size(new_grid, 2) - 1) |> Lux.f32

                st_kan.gen[symbol_map[i]].scale = [scale]
            end
        end

        z = Lux.apply(
            model.lkhood.generator.Φ_fcns[i],
            z,
            ps.gen.fcn[symbol_map[i]],
            st_kan.gen[symbol_map[i]],
        )
        z = dropdims(sum(z, dims = 1); dims = 1)
    end

    return ps, st_kan, st_lux
end

end
