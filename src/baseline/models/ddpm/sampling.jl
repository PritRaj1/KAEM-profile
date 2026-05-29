module DDPMSampling

export sample_loop

using Lux
using Reactant: @trace

using ..Utils
using ..DDPMModel: DDPM

function denoise_step(
        x,
        t_float,
        alpha,
        alpha_cumprod,
        alpha_cumprod_prev,
        posterior_variance,
        noise,
        noise_mask,
        unet,
        ps,
        st
    )
    # predict x_0, clamp [-1, 1], posterior mean
    noise_pred, st_new = unet(x, t_float, ps, st)

    sqrt_alpha_cumprod = sqrt.(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = sqrt.(1.0f0 .- alpha_cumprod)
    x0_pred = (x .- sqrt_one_minus_alpha_cumprod .* noise_pred) ./ sqrt_alpha_cumprod
    x0_pred = clamp.(x0_pred, -1.0f0, 1.0f0)

    one_minus_alpha_cumprod = 1.0f0 .- alpha_cumprod
    beta = 1.0f0 .- alpha
    coef_x0 = sqrt.(alpha_cumprod_prev) .* beta ./ one_minus_alpha_cumprod
    coef_xt = sqrt.(alpha) .* (1.0f0 .- alpha_cumprod_prev) ./ one_minus_alpha_cumprod
    mean = coef_x0 .* x0_pred .+ coef_xt .* x

    sigma = sqrt.(posterior_variance)
    x_prev = mean .+ sigma .* noise .* noise_mask
    return x_prev, st_new
end

function ddpm_step(
        i,
        x,
        st_curr,
        unet,
        ps,
        step_noise,
        timesteps,
        alphas,
        alphas_cumprod,
        alphas_cumprod_prev,
        posterior_variances,
        noise_masks,
        step_masks,
    )
    mask = step_masks[:, i]
    num_steps = size(step_masks, 1)
    t_mask = reshape(mask, 1, num_steps)
    t_float = dropdims(sum(timesteps .* t_mask; dims = 2); dims = 2)

    sched_mask = reshape(mask, 1, 1, 1, 1, num_steps)
    alpha = dropdims(sum(alphas .* sched_mask; dims = 5); dims = 5)
    alpha_cumprod_i = dropdims(sum(alphas_cumprod .* sched_mask; dims = 5); dims = 5)
    alpha_cumprod_prev_i = dropdims(sum(alphas_cumprod_prev .* sched_mask; dims = 5); dims = 5)
    posterior_variance = dropdims(sum(posterior_variances .* sched_mask; dims = 5); dims = 5)
    noise_mask = dropdims(sum(noise_masks .* sched_mask; dims = 5); dims = 5)

    noise = step_noise[:, :, :, :, i]

    x_new, st_new = denoise_step(
        x,
        t_float,
        alpha,
        alpha_cumprod_i,
        alpha_cumprod_prev_i,
        posterior_variance,
        noise,
        noise_mask,
        unet,
        ps,
        st_curr
    )
    return x_new, st_new
end

function sample_loop(
        unet,
        ps,
        st,
        st_rng,
        timesteps,
        alphas,
        alphas_cumprod,
        alphas_cumprod_prev,
        posterior_variances,
        betas,
        noise_masks,
        step_masks,
        num_steps::Int,
    )
    x_init = st_rng.x_init
    step_noise = st_rng.step_noise

    state = (1, x_init, st)
    @trace while first(state) <= num_steps
        i, x_curr, st_curr = state
        x_new, st_new = ddpm_step(
            i,
            x_curr,
            st_curr,
            unet,
            ps,
            step_noise,
            timesteps,
            alphas,
            alphas_cumprod,
            alphas_cumprod_prev,
            posterior_variances,
            noise_masks,
            step_masks,
        )
        state = (i + 1, x_new, st_new)
    end

    _, x_final, st_final = state
    x_clamped = clamp.(x_final, -1.0f0, 1.0f0)
    return x_clamped, st_final
end

end
