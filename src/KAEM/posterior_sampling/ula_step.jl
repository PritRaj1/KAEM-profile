module ULA_Step

export UlaKernel

using ..Utils
using ..KAEM_model

include("updates.jl")
using .LangevinUpdates

struct UlaKernel
    η
    sqrt_2η
    log_dist
    xchange_func
end

function (k::UlaKernel)(
        i,
        z_i,
        x_t,
        temps_gpu,
        model,
        ps,
        st_kan,
        st_lux,
        noise,
        log_u_swap,
        mask_swap_1,
        mask_swap_2,
        component_mask,
        temps,
    )
    ξ = noise[:, :, :, i]
    ∇z = unadjusted_grad(
        z_i,
        x_t,
        temps_gpu,
        model,
        ps,
        st_kan,
        st_lux,
        component_mask,
        k.log_dist,
    )
    new_z = z_i .+ k.η .* ∇z .+ k.sqrt_2η .* ξ

    return k.xchange_func(
        i,
        new_z,
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
end

end
