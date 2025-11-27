module ModelSetup

export prep_model

using ConfParser, Lux, Accessors, ComponentArrays, Random, Reactant, Enzyme, Optimisers

using ..Utils
using ..KAEM_model
using ..KAEM_model.LogPriorFCNs
using ..KAEM_model.InverseTransformSampling
using ..KAEM_model.PopulationXchange

include("train_steps/langevin_mle.jl")
include("train_steps/importance_sampling.jl")
include("train_steps/thermodynamic.jl")
include("ula/unadjusted_langevin.jl")
using .ImportanceSampling
using .LangevinMLE
using .ThermodynamicIntegration
using .ULA_sampling


function setup_training(
        opt_state,
        ps::ComponentArray{T},
        st_kan::ComponentArray{T},
        st_lux::NamedTuple,
        model::KAEM{T},
        x::AbstractArray{T};
        rng::AbstractRNG = Random.default_rng(),
        MLIR::Bool = true,
    ) where {T <: Float32}
    conf = model.conf

    # Posterior samplers
    initial_step_size =
        parse(Float32, retrieve(conf, "POST_LANGEVIN", "initial_step_size"))
    num_steps = parse(Int, retrieve(conf, "POST_LANGEVIN", "iters"))
    N_unadjusted = parse(Int, retrieve(conf, "POST_LANGEVIN", "N_unadjusted"))
    η_init = parse(Float32, retrieve(conf, "POST_LANGEVIN", "initial_step_size"))

    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
    zero_vec = pu(zeros(T, model.lkhood.x_shape..., model.batch_size, batch_size))
    max_samples = max(model.batch_size, batch_size)
    x = zeros(T, model.lkhood.x_shape..., max_samples) |> pu

    swap_replica_idxs = (
        model.N_t > 1 ?
            rand(rng, 1:(model.N_t - 1), num_steps) |> pu :
            nothing
    )

    # Prior sampling setup
    if model.prior.bool_config.ula
        num_steps_prior = parse(Int, retrieve(conf, "PRIOR_LANGEVIN", "iters"))
        step_size_prior = parse(Float32, retrieve(conf, "PRIOR_LANGEVIN", "step_size"))

        prior_sampler = initialize_ULA_sampler(
            model;
            η = step_size_prior,
            N = num_steps_prior,
            prior_sampling_bool = true,
        )
        @reset model.log_prior = LogPriorULA(model.ε)
        @reset model.sample_prior =
            (m, p, sk, sl, r) -> prior_sampler(p, sk, Lux.trainmode(sl), x; rng = r)

        println("Prior sampler: ULA")
    elseif model.prior.bool_config.mixture_model
        @reset model.sample_prior =
            (m, p, sk, sl, r) ->
        sample_mixture(m.prior, p.ebm, sk.ebm, Lux.testmode(sl.ebm), sk.quad; rng = r)

        @reset model.log_prior =
            LogPriorMix(model.ε, !model.prior.bool_config.contrastive_div)
        println("Prior sampler: Mix ITS")
    else
        @reset model.sample_prior =
            (m, p, sk, sl, r) ->
        sample_univariate(m.prior, p.ebm, sk.ebm, Lux.testmode(sl.ebm), sk.quad; rng = r)
        @reset model.log_prior =
            LogPriorUnivariate(model.ε, !model.prior.bool_config.contrastive_div)
        println("Prior sampler: Univar ITS")
    end

    # Default training criterion
    @reset model.train_step = begin

        static_loss = ImportanceLoss(model)

        if MLIR
            Reactant.@compile static_loss(
                opt_state,
                ps,
                st_kan,
                st_lux,
                x,
                1,
                rng,
                swap_replica_idxs
            )
        else
            static_loss
        end
    end

    @reset model.posterior_sampler = initialize_ULA_sampler(
        model;
        η = η_init,
        N = num_steps,
    )

    if model.N_t > 1

        Q, S = model.prior.q_size, model.batch_size
        P = model.prior.bool_config.mixture_model ? 1 : model.prior.p_size
        @reset model.xchange_func = ReplicaXchange(Q, P, S, model.N_t)

        @reset model.train_step = begin

            static_loss = ThermoLoss(model)

            if MLIR
                Reactant.@compile static_loss(
                    opt_state,
                    ps,
                    st_kan,
                    st_lux,
                    x,
                    1,
                    rng,
                    swap_replica_idxs
                )
            else
                static_loss
            end
        end
        println("Posterior sampler: Thermo ULA")

    elseif model.MALA || model.prior.bool_config.ula

        static_loss = LangevinLoss(model)

        @reset model.train_step = begin
            if MLIR
                Reactant.@compile static_loss(
                    opt_state,
                    ps,
                    st_kan,
                    st_lux,
                    x,
                    1,
                    rng,
                    swap_replica_idxs
                )
            else
                static_loss
            end
        end
    else

        println("Posterior sampler: MLE IS")
    end

    return model
end

function prep_model(
        model::KAEM{T},
        x::AbstractArray{T},
        optimizer;
        rng::AbstractRNG = Random.default_rng(),
        MLIR::Bool = true,
    ) where {T <: Float32}
    ps = Lux.initialparameters(rng, model)
    st_kan, st_lux = Lux.initialstates(rng, model)
    ps, st_kan, st_lux =
        ps |> ComponentArray |> Lux.f32 |> pu, st_kan |> ComponentArray |> Lux.f32 |> pu, st_lux |> Lux.f32 |> pu
    opt_state = Optimisers.setup(optimizer.rule(), ps)
    model = setup_training(
        opt_state,
        ps,
        st_kan,
        st_lux,
        model::KAEM{T},
        x;
        rng = rng,
        MLIR = MLIR
    )
    return model, opt_state, ps, st_kan, st_lux
end

end
