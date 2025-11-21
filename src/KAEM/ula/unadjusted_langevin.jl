module ULA_sampling

export initialize_ULA_sampler, ULA_sampler

using Reactant: @trace
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

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_MALA

π_dist = Dict(
    "uniform" => (p, b, rng) -> rand(rng, Float32, p, 1, b),
    "gaussian" => (p, b, rng) -> randn(rng, Float32, p, 1, b),
    "lognormal" => (p, b, rng) -> rand(rng, LogNormal(0.0f0, 1.0f0), p, 1, b),
    "ebm" => (p, b, rng) -> randn(rng, Float32, p, 1, b),
)

struct ULA_sampler{T}
    prior_sampling_bool::Bool
    N::Int
    RE_frequency::Int
    η::T
    model::KAEM{T}
end

function initialize_ULA_sampler(
        model::KAEM{T};
        η::T = 1.0f-3,
        prior_sampling_bool::Bool = false,
        N::Int = 20,
        RE_frequency::Int = 10,
    ) where {T}

    return ULA_sampler(prior_sampling_bool, N, RE_frequency, η, model)
end

function (sampler::ULA_sampler)(
        ps,
        st_kan,
        st_lux,
        x;
        temps = [1.0f0],
        rng = Random.default_rng(),
        swap_replica_idxs = nothing,
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
    model = sampler.model
    η = sampler.η
    sqrt_2η = sqrt(2 * η)
    seq = model.lkhood.SEQ
    Q, S = model.prior.q_size, model.batch_size
    P = model.prior.bool_config.mixture_model ? 1 : model.prior.p_size
    num_temps = (model.N_t > 1 && !sampler.prior_sampling_bool) ? model.N_t : 1

    # Initialize from prior
    z = begin
        if model.prior.bool_config.ula && sampler.prior_sampling_bool
            π_dist[model.prior.prior_type](P, S, rng)
        else
            z_initial, st_ebm =
                model.sample_prior(model, ps, st_kan, st_lux, rng)
            z_samples = Vector{typeof(z_initial)}()
            sizehint!(z_samples, length(temps))
            push!(z_samples, z_initial)

            for i in 1:(length(temps) - 1)
                z_i, st_ebm =
                    model.sample_prior(model, ps, st_kan, st_lux, rng)
                push!(z_samples, z_i)
            end
            @reset st_lux.ebm = st_ebm
            cat(z_samples..., dims = 3)
        end
    end

    z = reshape(z, Q, P, S, num_temps) .* 1.0f0
    log_dist = sampler.prior_sampling_bool ? unadjusted_logprior : unadjusted_logpos

    N_steps = sampler.N
    RE_freq = sampler.RE_frequency

    # Pre-allocate noise
    noise = randn(rng, Float32, Q, P, S, num_temps, N_steps)
    log_u_swap = log.(rand(rng, Float32, num_temps - 1, N_steps))
    ll_noise = randn(rng, Float32, model.lkhood.x_shape..., S, 2, num_temps, N_steps)

    @trace for i in 1:N_steps
        ξ = noise[:, :, :, :, i]
        ∇z =
            unadjusted_grad(
            z,
            x,
            temps,
            model,
            ps,
            st_kan,
            st_lux,
            num_temps,
            log_dist,
        )

        @. z = z + η * ∇z + sqrt_2η * ξ

        if i % RE_freq == 0 && num_temps > 1
            t = swap_replica_idxs[i] # Randomly pick two adjacent temperatures to swap
            z_t = z[:, :, :, t]
            z_t1 = z[:, :, :, t + 1]

            noise_1 =
                model.lkhood.SEQ ? ll_noise[:, :, :, 1, t, i] :
                ll_noise[:, :, :, :, 1, t, i]
            noise_2 =
                model.lkhood.SEQ ? ll_noise[:, :, :, 2, t, i] :
                ll_noise[:, :, :, :, 2, t, i]

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
            swap = log_u_swap[t:t, i:i] .< log_swap_ratio
            @reset st_lux.gen = st_gen

            z[:, :, :, t] .= swap .* z_t1 .+ (1 .- swap) .* z_t
            z[:, :, :, t + 1] .= (1 .- swap) .* z_t1 .+ swap .* z_t
        end
    end

    if prior_sampling
        st_lux = st_lux.ebm
        z = dropdims(z; dims = 4)
    end

    return z, st_lux
end


end
