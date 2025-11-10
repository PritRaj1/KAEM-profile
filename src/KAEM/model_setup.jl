module ModelSetup

export prep_model

using ConfParser, Lux, Accessors, ComponentArrays, Random, Reactant, Enzyme

using ..Utils
using ..KAEM_model
using ..KAEM_model.LogPriorFCNs
using ..KAEM_model.InverseTransformSampling

include("loss_fcns/langevin_mle.jl")
include("loss_fcns/importance_sampling.jl")
include("loss_fcns/thermodynamic.jl")
include("ula/unadjusted_langevin.jl")
using .ImportanceSampling
using .LangevinMLE
using .ThermodynamicIntegration
using .ULA_sampling


function setup_training(
        ps::ComponentArray{T},
        st_kan::ComponentArray{T},
        st_lux::NamedTuple,
        model::KAEM{T},
        x::AbstractArray{T};
        rng::AbstractRNG = Random.default_rng()
    ) where {T <: Float32}
    conf = model.conf

    # Posterior samplers
    initial_step_size =
        parse(Float32, retrieve(conf, "POST_LANGEVIN", "initial_step_size"))
    num_steps = parse(Int, retrieve(conf, "POST_LANGEVIN", "iters"))
    N_unadjusted = parse(Int, retrieve(conf, "POST_LANGEVIN", "N_unadjusted"))
    η_init = parse(Float32, retrieve(conf, "POST_LANGEVIN", "initial_step_size"))
    replica_exchange_frequency = parse(
        Int,
        retrieve(conf, "THERMODYNAMIC_INTEGRATION", "replica_exchange_frequency"),
    )

    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
    zero_vec = pu(zeros(T, model.lkhood.x_shape..., model.IS_samples, batch_size))
    max_samples = max(model.IS_samples, batch_size)
    x = zeros(T, model.lkhood.x_shape..., max_samples) |> pu

    # Defaults
    @reset model.loss_fcn = Reactant.@compile importance_loss(
        ps,
        st_kan,
        st_lux,
        model,
        x;
        train_idx = 1,
        rng = rng
    )

    @reset model.posterior_sampler = initialize_ULA_sampler(;
        η = η_init,
        N = num_steps,
        RE_frequency = replica_exchange_frequency,
    )

    if model.N_t > 1
        @reset model.loss_fcn = Reactant.@compile thermodynamic_loss(
            ps,
            st_kan,
            st_lux,
            model,
            x;
            train_idx = 1,
            rng = rng
        )

        println("Posterior sampler: Thermo ULA")
    elseif model.MALA || model.prior.bool_config.ula
        @reset model.loss_fcn = Reactant.@compile langevin_loss(
            ps,
            st_kan,
            st_lux,
            model,
            x;
            train_idx = 1,
            rng = rng
        )
        println("Posterior sampler: MLE ULA")
    else

        println("Posterior sampler: MLE IS")
    end

    if model.prior.bool_config.ula
        num_steps_prior = parse(Int, retrieve(conf, "PRIOR_LANGEVIN", "iters"))
        step_size_prior = parse(Float32, retrieve(conf, "PRIOR_LANGEVIN", "step_size"))

        prior_sampler = initialize_ULA_sampler(;
            η = step_size_prior,
            N = num_steps_prior,
            prior_sampling_bool = true,
        )

        @reset model.sample_prior =
            (m, n, p, sk, sl, r) -> prior_sampler(m, p, sk, Lux.testmode(sl), x; rng = r)

        @reset model.log_prior = LogPriorULA(model.ε)
        println("Prior sampler: ULA")
    elseif model.prior.bool_config.mixture_model
        @reset model.sample_prior =
            (m, n, p, sk, sl, r) ->
        sample_mixture(m.prior, n, p.ebm, sk.ebm, sl.ebm; rng = r)

        @reset model.log_prior =
            LogPriorMix(model.ε, !model.prior.bool_config.contrastive_div)
        println("Prior sampler: Mix ITS, Quadrature method: $(model.prior.quad_type)")
    else
        @reset model.sample_prior =
            (m, n, p, sk, sl, r) ->
        sample_univariate(m.prior, n, p.ebm, sk.ebm, sl.ebm; rng = r)
        @reset model.log_prior =
            LogPriorUnivariate(model.ε, !model.prior.bool_config.contrastive_div)
        println("Prior sampler: Univar ITS, Quadrature method: $(model.prior.quad_type)")
    end

    return model
end

function prep_model(
        model::KAEM{T},
        x::AbstractArray{T};
        rng::AbstractRNG = Random.default_rng(),
    ) where {T <: Float32}
    ps = Lux.initialparameters(rng, model)
    st_kan, st_lux = Lux.initialstates(rng, model)
    ps, st_kan, st_lux =
        ps |> ComponentArray |> Lux.f32 |> pu, st_kan |> ComponentArray |> Lux.f32 |> pu, st_lux |> Lux.f32 |> pu
    model = setup_training(ps, st_kan, st_lux, model::KAEM{T}, x; rng = rng)
    return model, ps, st_kan, st_lux
end

end
