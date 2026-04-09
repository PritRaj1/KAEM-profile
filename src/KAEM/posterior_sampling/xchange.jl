module PopulationXchange

export ReplicaXchange, NoExchange

using LinearAlgebra

using ..Utils
using ..KAEM_model

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_MALA

struct _ReplicaXchange
    Q::Int
    P::Int
    S::Int
    num_temps::Int
    shift_up
    shift_down
    ll_diff_mat
end

function ReplicaXchange(Q::Int, P::Int, S::Int, T::Int)
    shift_up = zeros(Float32, 1, 1, 1, T, T)
    shift_down = zeros(Float32, 1, 1, 1, T, T)
    for t in 1:(T - 1)
        shift_up[1, 1, 1, t + 1, t] = 1.0f0
        shift_down[1, 1, 1, t, t + 1] = 1.0f0
    end
    ll_diff_mat = shift_up[1, 1, 1, :, :] - Matrix{Float32}(I, T, T)
    return _ReplicaXchange(
        Q,
        P,
        S,
        T,
        shift_up,
        shift_down,
        ll_diff_mat
    )
end

function (r::_ReplicaXchange)(
        i,
        z_i,
        x_t,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
        log_u_swap,
        mask_swap_1,
        mask_swap_2,
    )
    Q, P, S, T = r.Q, r.P, r.S, r.num_temps
    ll_all = first(
        log_likelihood_MALA(
            z_i,
            x_t,
            model.lkhood,
            ps.gen,
            st_kan.gen,
            st_lux.gen,
            zero(x_t);
            ε = model.ε,
        )
    )

    ll_st = reshape(ll_all, S, T)
    mask1 = mask_swap_1[:, :, i]
    mask2 = mask_swap_2[:, :, i]

    # Fused shift-subtract: ll_diff[s,t] = ll[s,t+1] - ll[s,t]
    ll_diff = ll_st * r.ll_diff_mat
    temps_row = reshape(temps, 1, T)
    temps_diff = temps_row .- reshape(r.shift_down[1, 1, 1, :, :] * temps, 1, T)
    ratio = mask1 .* temps_diff .* ll_diff

    log_u = log_u_swap[:, :, i]
    accept = mask1 .* max.(sign.(ratio .- log_u), 0.0f0)

    # Shift accept up: accept_upper[t+1] = accept[t]
    accept_upper = (accept * r.shift_down[1, 1, 1, :, :]) .* mask2

    z = reshape(z_i, Q, P, S, T)
    z_down = dropdims(sum(z .* r.shift_up, dims = 4); dims = 4)
    z_up = dropdims(sum(z .* r.shift_down, dims = 4); dims = 4)

    accept_exp = reshape(accept, 1, 1, S, T)
    accept_upper_exp = reshape(accept_upper, 1, 1, S, T)
    mask1_exp = reshape(mask1, 1, 1, 1, T)
    mask2_exp = reshape(mask2, 1, 1, 1, T)

    z_new = (
        mask1_exp .* (accept_exp .* z_down .+ (1.0f0 .- accept_exp) .* z) .+
            mask2_exp .* (accept_upper_exp .* z_up .+ (1.0f0 .- accept_upper_exp) .* z) .+
            (1.0f0 .- mask1_exp .- mask2_exp) .* z
    )

    return reshape(z_new, Q, P, S * T)
end

struct NoExchange end

function (r::NoExchange)(
        i,
        z_i,
        x_t,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
        log_u_swap,
        mask_swap_1,
        mask_swap_2,
    )
    return z_i
end

end
