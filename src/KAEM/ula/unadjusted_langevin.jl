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

π_dist = Dict(
    "uniform" => (p, b, rng) -> rand(rng, Float32, p, 1, b),
    "gaussian" => (p, b, rng) -> randn(rng, Float32, p, 1, b),
    "lognormal" => (p, b, rng) -> rand(rng, LogNormal(0.0f0, 1.0f0), p, 1, b),
    "ebm" => (p, b, rng) -> randn(rng, Float32, p, 1, b),
)

struct ULA_sampler
    prior_sampling_bool
    N
    η
    sqrt_2η
    model
    Q
    P
    S
    num_temps
    thermo_bool
    log_dist
end

function initialize_ULA_sampler(
        model::KAEM{T};
        η::T = 1.0f-3,
        prior_sampling_bool::Bool = false,
        N::Int = 20,
    ) where {T}

    Q, S = model.prior.q_size, model.batch_size
    P = model.prior.bool_config.mixture_model ? 1 : model.prior.p_size
    num_temps = (model.N_t > 1 && !prior_sampling_bool) ? model.N_t : 1
    thermo_bool = num_temps > 1
    log_dist = prior_sampling_bool ? unadjusted_logprior : unadjusted_logpos

    return ULA_sampler(
        prior_sampling_bool,
        N,
        η,
        sqrt(2 * η),
        model,
        Q,
        P,
        S,
        num_temps,
        thermo_bool,
        log_dist,
    )
end

function step(
        i,
        z_i,
        x,
        x_t,
        temps,
        temps_gpu,
        η,
        sqrt_2η,
        sampler,
        model,
        lkhood_copy,
        ps,
        st_kan,
        st_lux,
        noise,
        log_u_swap,
        ll_noise,
        mask_swap_1,
        mask_swap_2,
    )
    ξ = noise[:, :, :, i]
    ∇z =
        unadjusted_grad(
        z_i,
        x_t,
        temps_gpu,
        model,
        ps,
        st_kan,
        st_lux,
        sampler.log_dist,
    )
    new_z = z_i .+ η .* ∇z .+ sqrt_2η .* ξ
    return model.xchange_func(
        i,
        new_z,
        x,
        temps,
        model,
        lkhood_copy,
        ps,
        st_kan,
        st_lux,
        log_u_swap,
        ll_noise,
        mask_swap_1,
        mask_swap_2,
    )
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
    sqrt_2η = sampler.sqrt_2η
    seq = model.lkhood.SEQ
    Q, P, S, num_temps = sampler.Q, sampler.P, sampler.S, sampler.num_temps
    thermo_bool = sampler.thermo_bool
    prior_sampling_bool = sampler.prior_sampling_bool

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

    N_steps = sampler.N
    x_t = !prior_sampling_bool ? (
        model.lkhood.SEQ ? repeat(x, 1, 1, num_temps) :
            (model.use_pca ? repeat(x, 1, num_temps) : repeat(x, 1, 1, 1, num_temps))
    ) : nothing

    # Pre-allocate noise
    noise = randn(rng, Float32, Q, P, S * num_temps, N_steps)
    log_u_swap = log.(rand(rng, Float32, num_temps, N_steps))
    ll_noise = randn(rng, Float32, model.lkhood.x_shape..., S, 2, num_temps, N_steps)
    swap_replica_idxs_plus = isnothing(swap_replica_idxs) ? nothing : swap_replica_idxs .+ 1

    # Traced HLO does not support int arrays, so handle mask outside
    mask_swap_1 = isnothing(swap_replica_idxs) ? nothing : Lux.f32(1:num_temps .== swap_replica_idxs') .* 1.0f0
    mask_swap_2 = isnothing(swap_replica_idxs) ? nothing : Lux.f32(1:num_temps .== swap_replica_idxs_plus') .* 1.0f0

    i = 1
    @trace while i <= N_steps
        z_flat = step(
            i,
            z_flat,
            x,
            x_t,
            temps,
            temps_gpu,
            η,
            sqrt_2η,
            sampler,
            model,
            lkhood_copy,
            ps,
            st_kan,
            st_lux,
            noise,
            log_u_swap,
            ll_noise,
            mask_swap_1,
            mask_swap_2,
        )
        i += 1
    end

    z = reshape(z_flat, Q, P, S, num_temps)

    if prior_sampling_bool
        st_lux = st_lux.ebm
        z = dropdims(z; dims = 4)
    end

    return z, st_lux
end


end
