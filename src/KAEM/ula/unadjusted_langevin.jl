module ULA_sampling

export initialize_ULA_sampler, ULA_sampler

using LinearAlgebra,
    Random,
    Lux,
    Distributions,
    Accessors,
    Statistics,
    ComponentArrays

using ..Utils
using ..KAEM_model

include("updates.jl")
using .LangevinUpdates

π_dist = Dict(
    "uniform" => (p, b, rng) -> rand(rng, Float32, p, 1, b),
    "gaussian" => (p, b, rng) -> randn(rng, Float32, p, 1, b),
    "lognormal" => (p, b, rng) -> rand(rng, LogNormal(0.0f0, 1.0f0), p, 1, b),
    "ebm" => (p, b, rng) -> randn(rng, Float32, p, 1, b),
)

struct ULA_sampler{T <: Float32}
    prior_sampling_bool::Bool
    N::Int
    RE_frequency::Int
    η::T
end

function initialize_ULA_sampler(;
        η::T = 1.0f-3,
        prior_sampling_bool::Bool = false,
        N::Int = 20,
        RE_frequency::Int = 10,
    ) where {T <: Float32}

    return ULA_sampler(prior_sampling_bool, N, RE_frequency, η)
end

function (sampler::ULA_sampler)(
        model,
        ps,
        st_kan,
        st_lux,
        x;
        temps = [1.0f0],
        rng = Random.default_rng(),
    )
    """
    Unadjusted Langevin Algorithm (ULA) sampler to generate posterior samples.

    Args:
        m: The model.
        ps: The parameters of the model.
        st: The states of the model.
        x: The data.
        t: The temperatures if using Thermodynamic Integration.
        N: The number of iterations.
        rng: The random number generator.

        
    Unused arguments:
        N_unadjusted: The number of unadjusted iterations.
        Δη: The step size increment.
        η_min: The minimum step size.
        η_max: The maximum step size.

    Returns:
        The posterior samples.
    """
    # Initialize from prior
    z_hq = begin
        if model.prior.bool_config.ula && sampler.prior_sampling_bool
            z = π_dist[model.prior.prior_type](model.prior.p_size, size(x)[end], rng)
            pu(z)
        else
            z_initial, st_ebm =
                model.sample_prior(model, size(x)[end], ps, st_kan, st_lux, rng)
            z_samples = Vector{typeof(z_initial)}()
            sizehint!(z_samples, length(temps))
            push!(z_samples, z_initial)

            for i in 1:(length(temps) - 1)
                z_i, st_ebm =
                    model.sample_prior(model, size(x)[end], ps, st_kan, st_lux, rng)
                push!(z_samples, z_i)
            end
            @reset st_lux.ebm = st_ebm
            cat(z_samples..., dims = 3)
        end
    end

    η = sampler.η
    sqrt_2η = sqrt(2 * η)
    seq = model.lkhood.SEQ

    num_temps, Q, P, S = length(temps), size(z_hq)[1:2]..., size(x)[end]
    S = sampler.prior_sampling_bool ? size(z_hq)[end] : S
    z_hq = reshape(z_hq, Q, P, S, num_temps)
    temps_gpu = repeat(temps, S)

    # Pre-allocate for both precisions
    z_fq = reshape(z_hq, Q, P, S * num_temps)
    ∇z_fq = 0.0f0 .* z_fq
    z_copy = similar(z_hq[:, :, :, 1])
    z_t, z_t1 = z_copy, z_copy

    x_t = (
        model.lkhood.SEQ ? repeat(x, 1, 1, num_temps) :
            (model.use_pca ? repeat(x, 1, num_temps) : repeat(x, 1, 1, 1, num_temps))
    )

    # Pre-allocate noise
    noise = randn(rng, Float32, Q, P, S * num_temps, sampler.N)
    log_u_swap = log.(rand(rng, Float32, num_temps - 1, sampler.N))
    ll_noise = randn(rng, Float32, model.lkhood.x_shape..., S, 2, num_temps, sampler.N)
    swap_replica_idxs = num_temps > 1 ? rand(rng, 1:(num_temps - 1), sampler.N) : nothing

    for i in 1:sampler.N
        ξ = @view(noise[:, :, :, i])
        ∇z_fq .=
            unadjusted_logpos_grad(
            z_fq,
            x_t,
            temps_gpu,
            model,
            ps,
            st_kan,
            st_lux,
            sampler.prior_sampling_bool,
        )

        update_z!(z_fq, ∇z_fq, η, ξ, sqrt_2η, Q, P, S)
        z_hq .= (reshape(z_fq, Q, P, S, num_temps))

        if i % sampler.RE_frequency == 0 && num_temps > 1 && !sampler.prior_sampling_bool
            t = swap_replica_idxs[i] # Randomly pick two adjacent temperatures to swap
            z_t = @view(z_hq[:, :, :, t])
            z_t1 = @view(z_hq[:, :, :, t + 1])

            noise_1 =
                model.lkhood.SEQ ? @view(ll_noise[:, :, :, 1, t, i]) :
                @view(ll_noise[:, :, :, :, 1, t, i])
            noise_2 =
                model.lkhood.SEQ ? @view(ll_noise[:, :, :, 2, t, i]) :
                @view(ll_noise[:, :, :, :, 2, t, i])

            ll_t, st_gen = log_likelihood_MALA(
                z_t,
                x,
                model.lkhood,
                ps.gen,
                st_kan.gen,
                st_lux.gen,
                noise_1;
                ε = model.ε,
            )
            ll_t1, st_gen = log_likelihood_MALA(
                z_t1,
                x,
                model.lkhood,
                ps.gen,
                st_kan.gen,
                st_lux.gen,
                noise_2;
                ε = model.ε,
            )

            log_swap_ratio = (temps[t + 1] - temps[t]) .* (sum(ll_t) - sum(ll_t1))
            swap = T(log_u_swap[t, i] < log_swap_ratio)
            @reset st_lux.gen = st_gen

            z_hq[:, :, :, t] .= swap .* z_t1 .+ (1 - swap) .* z_t
            z_hq[:, :, :, t + 1] .= (1 - swap) .* z_t1 .+ swap .* z_t
            z_fq .= (reshape(z_hq, Q, P, S * num_temps))
        end
    end

    if sampler.prior_sampling_bool
        st_lux = st_lux.ebm
        z_hq = dropdims(z_hq; dims = 4)
    end

    return z_hq, st_lux
end


end
