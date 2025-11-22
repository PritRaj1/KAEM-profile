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

    thermo_bool = model.N_t > 1
    num_temps = (thermo_bool && !sampler.prior_sampling_bool) ? model.N_t : 1

    # Initialize from prior
    z_flat = begin
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

    lkhood_copy = deepcopy(model.lkhood)

    for i in 1:model.prior.depth
        @reset model.prior.fcns_qp[i].basis_function.S = S * num_temps
    end

    if !model.lkhood.SEQ && !model.lkhood.CNN
        for i in 1:model.lkhood.generator.depth
            @reset model.lkhood.generator.Φ_fcns[i].basis_function.S = S * num_temps
        end
    end

    @reset model.prior.s_size = S * num_temps
    @reset model.lkhood.generator.s_size = S * num_temps

    temps_gpu = repeat(temps, S)
    log_dist = sampler.prior_sampling_bool ? unadjusted_logprior : unadjusted_logpos

    N_steps = sampler.N
    RE_freq = sampler.RE_frequency
    x_t = (
        model.lkhood.SEQ ? repeat(x, 1, 1, num_temps) :
            (model.use_pca ? repeat(x, 1, num_temps) : repeat(x, 1, 1, 1, num_temps))
    )

    # Pre-allocate noise
    noise = randn(rng, Float32, Q, P, S * num_temps, N_steps)
    log_u_swap = log.(rand(rng, Float32, num_temps, N_steps))
    ll_noise = randn(rng, Float32, model.lkhood.x_shape..., S, 2, num_temps, N_steps)
    swap_replica_idxs_plus = isnothing(swap_replica_idxs) ? nothing : swap_replica_idxs .+ 1

    # Traced HLO does not support int arrays, so handle mask outside
    mask_swap_1 = Lux.f32(1:num_temps .== swap_replica_idxs') .* 1.0f0
    mask_swap_2 = Lux.f32(1:num_temps .== swap_replica_idxs_plus') .* 1.0f0

    function replica_exchange_step(z_i, i)
        z = reshape(z_i, Q, P, S, num_temps)

        # Randomly pick two adjacent temperatures to swap
        mask1 = mask_swap_1[:, i]
        mask2 = mask_swap_2[:, i]
        z_t = dropdims(sum(z .* reshape(mask1, 1, 1, 1, num_temps), dims = 4); dims = 4)
        z_t1 = dropdims(sum(z .* reshape(mask2, 1, 1, 1, num_temps), dims = 4); dims = 4)

        noise_1 =
            (
            model.lkhood.SEQ ?
                dropdims(
                    sum(
                        ll_noise[:, :, :, 1, :, i] .* reshape(mask1, 1, 1, 1, num_temps)
                        , dims = 4
                    ); dims = 4
                ) :
                dropdims(
                    sum(
                        ll_noise[:, :, :, :, 1, :, i] .* reshape(mask1, 1, 1, 1, 1, num_temps)
                        , dims = 5
                    ); dims = 5
                )
        )
        noise_2 =
            (
            model.lkhood.SEQ ?
                dropdims(
                    sum(
                        ll_noise[:, :, :, 2, :, i] .* reshape(mask2, 1, 1, 1, num_temps)
                        , dims = 4
                    ); dims = 4
                ) :
                dropdims(
                    sum(
                        ll_noise[:, :, :, :, 2, :, i] .* reshape(mask2, 1, 1, 1, 1, num_temps)
                        , dims = 5
                    ); dims = 5
                )
        )

        ll_t, st_gen = log_likelihood_MALA(
            z_t,
            x,
            lkhood_copy,
            ps.gen,
            st_kan.gen,
            st_lux.gen,
            noise_1;
            ε = model.ε,
        )
        ll_t1, st_gen = log_likelihood_MALA(
            z_t1,
            x,
            lkhood_copy,
            ps.gen,
            st_kan.gen,
            st_lux.gen,
            noise_2;
            ε = model.ε,
        )

        # Global exchange criterion
        temps_t = sum(temps .* mask1)
        temps_t1 = sum(temps .* mask2)
        log_swap_ratio = (temps_t1 - temps_t) .* (sum(ll_t) - sum(ll_t1))
        swap = sum(log_u_swap[:, i] .* mask1) < log_swap_ratio
        @reset st_lux.gen = st_gen

        z = (
            (swap .* z_t1 .+ (1 .- swap) .* z_t) .+ # Swap or not
                reshape(mask1, 1, 1, 1, num_temps) .* z # Index of t
        )
        z = (
            ((1 .- swap) .* z_t1 .+ swap .* z_t) .+ # Swap or not
                reshape(mask2, 1, 1, 1, num_temps) .* z # Index of t1
        )

        return reshape(z, Q, P, S * num_temps)
    end

    RE_func = isnothing(swap_replica_idxs) ? (z_i, i) -> z_i : replica_exchange_step

    @trace for i in 1:N_steps
        ξ = noise[:, :, :, i]
        ∇z =
            unadjusted_grad(
            z_flat,
            x_t,
            temps_gpu,
            model,
            ps,
            st_kan,
            st_lux,
            log_dist,
        )

        @. z_flat += η * ∇z + sqrt_2η * ξ
        z_flat = ifelse(i % RE_freq == 0, RE_func(z_flat, i), z_flat)
    end

    z = reshape(z_flat, Q, P, S, num_temps)

    if prior_sampling
        st_lux = st_lux.ebm
        z = dropdims(z; dims = 4)
    end

    return z, st_lux
end


end
