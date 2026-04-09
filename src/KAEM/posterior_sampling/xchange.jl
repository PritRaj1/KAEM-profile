module PopulationXchange

export ReplicaXchange, NoExchange

using ..Utils
using ..KAEM_model

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_MALA

struct ReplicaXchange
    Q::Int
    P::Int
    S::Int
    num_temps::Int
end

function (r::ReplicaXchange)(
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
    Q, P, S, num_temps = r.Q, r.P, r.S, r.num_temps
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

    ll_st = reshape(ll_all, S, num_temps)
    mask1 = mask_swap_1[:, :, i]
    mask2 = mask_swap_2[:, :, i]

    # Shift via slice+pad
    pad_s = ll_st[:, 1:1] .* 0.0f0
    ll_shifted = cat(ll_st[:, 2:end], pad_s; dims = 2)

    temps_row = reshape(temps, 1, num_temps)
    pad_t = temps_row[:, 1:1] .* 0.0f0
    temps_shifted = cat(temps_row[:, 2:end], pad_t; dims = 2)

    # Accept/reject
    ratio = mask1 .* (temps_row .- temps_shifted) .* (ll_shifted .- ll_st)
    log_u = log_u_swap[:, :, i]
    accept = mask1 .* max.(sign.(ratio .- log_u), 0.0f0)

    # Shift accept up: accept_upper[t+1] = accept[t]
    pad_a = accept[:, 1:1] .* 0.0f0
    accept_upper = cat(pad_a, accept[:, 1:(end - 1)]; dims = 2) .* mask2

    z = reshape(z_i, Q, P, S, num_temps)
    z_flat_temps = reshape(z, Q * P * S, num_temps)

    # z_down[t] = z[t+1], z_up[t] = z[t-1]
    pad_z = z_flat_temps[:, 1:1] .* 0.0f0
    z_down = reshape(
        cat(z_flat_temps[:, 2:end], pad_z; dims = 2),
        Q, P, S, num_temps,
    )
    z_up = reshape(
        cat(pad_z, z_flat_temps[:, 1:(end - 1)]; dims = 2),
        Q, P, S, num_temps,
    )

    accept_exp = reshape(accept, 1, 1, S, num_temps)
    accept_upper_exp = reshape(accept_upper, 1, 1, S, num_temps)
    mask1_exp = reshape(mask1, 1, 1, 1, num_temps)
    mask2_exp = reshape(mask2, 1, 1, 1, num_temps)

    z_new = (
        mask1_exp .* (accept_exp .* z_down .+ (1.0f0 .- accept_exp) .* z) .+
            mask2_exp .* (accept_upper_exp .* z_up .+ (1.0f0 .- accept_upper_exp) .* z) .+
            (1.0f0 .- mask1_exp .- mask2_exp) .* z
    )

    return reshape(z_new, Q, P, S * num_temps) .* 1.0f0
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
