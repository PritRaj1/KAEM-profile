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
include("train_steps/variational.jl")
include("posterior_sampling/ula.jl")
include("posterior_sampling/pcnl.jl")
include("rng.jl")
using .ImportanceSampling
using .LangevinMLE
using .ThermodynamicIntegration
using .VariationalTraining
using .ULA_sampling
using .pCNL_sampling
using .HLOrng

# Compile or return raw loss function
function maybe_compile(loss, MLIR, opt_state, ps, st_kan, st_lux, x, st_rng)
    return MLIR ? Reactant.@compile(loss(opt_state, ps, st_kan, st_lux, x, 1, st_rng)) : loss
end

function setup_training(
        opt_state,
        ps::ComponentArray{T},
        st_kan::NamedTuple,
        st_lux::NamedTuple,
        model::KAEM{T},
        x::AbstractArray{T};
        rng::AbstractRNG = Random.MersenneTwister(1),
        MLIR::Bool = true,
    ) where {T <: Float32}
    conf = model.conf

    num_steps = parse(Int, retrieve(conf, "POST_LANGEVIN", "iters"))
    η = parse(Float32, retrieve(conf, "POST_LANGEVIN", "ula_eta"))

    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
    max_samples = max(model.batch_size, batch_size)
    x = zeros(T, model.lkhood.x_shape..., max_samples) |> pu

    # Prior sampling setup
    if model.prior.bool_config.ula
        num_steps_prior = parse(Int, retrieve(conf, "PRIOR_LANGEVIN", "iters"))
        step_size_prior = parse(Float32, retrieve(conf, "PRIOR_LANGEVIN", "step_size"))

        prior_sampler = initialize_pCNL_sampler(
            model;
            δ = step_size_prior,
            N = num_steps_prior,
            prior_sampling_bool = true,
        )
        @reset model.log_prior = LogPriorULA(model.ε)
        @reset model.sample_prior =
            (m, p, sk, sl, r) -> (out = prior_sampler(p, sk, Lux.trainmode(sl), x, r); (out[1], out[2]))

        println("Prior sampler: pCNL")
    elseif model.prior.bool_config.mixture_model
        @reset model.sample_prior =
            (m, p, sk, sl, r) ->
        sample_mixture(m.prior, p.ebm, sk.ebm, Lux.testmode(sl.ebm), sk.quad, r)
        @reset model.log_prior =
            LogPriorMix(model.ε, !model.prior.bool_config.contrastive_div)
        println("Prior sampler: Mix ITS")
    else
        @reset model.sample_prior =
            (m, p, sk, sl, r) ->
        sample_univariate(m.prior, p.ebm, sk.ebm, Lux.testmode(sl.ebm), sk.quad, r)
        @reset model.log_prior =
            LogPriorUnivariate(model.ε, !model.prior.bool_config.contrastive_div)
        println("Prior sampler: Univar ITS")
    end

    # Posterior sampler setup
    exchange_type = retrieve(conf, "THERMODYNAMIC_INTEGRATION", "exchange_type")
    if model.sampler_type == "pcnl"
        δ = parse(Float32, retrieve(conf, "POST_LANGEVIN", "pcnl_delta"))
        @reset model.posterior_sampler = initialize_pCNL_sampler(
            model; δ = δ, N = num_steps, exchange_type = exchange_type,
        )
    elseif model.sampler_type == "ula"
        @reset model.posterior_sampler = initialize_ULA_sampler(
            model; η = η, N = num_steps, exchange_type = exchange_type,
        )
    else
        @reset model.posterior_sampler = initialize_pCNL_sampler(model; N = num_steps)
    end

    # Init per-temperature step sizes
    num_temps = model.posterior_sampler.num_temps
    δ_init = model.sampler_type == "pcnl" ?
        parse(Float32, retrieve(conf, "POST_LANGEVIN", "pcnl_delta")) :
        (model.sampler_type == "ula" ? η : 0.01f0)
    @reset st_lux.delta = pu(fill(δ_init, num_temps))

    st_rng = seed_rand(model; rng = rng)

    # Forward pass to init st_lux state before compilation
    _, st_ebm, st_gen = Reactant.@jit model(ps, st_kan, Lux.trainmode(st_lux), st_rng)
    @reset st_lux.ebm = st_ebm
    @reset st_lux.gen = st_gen
    @reset st_lux.delta = Reactant.@jit adapt_delta(st_lux.delta, st_lux.delta, 1)

    num_param_updates =
        parse(Int, retrieve(conf, "TRAINING", "N_epochs")) * length(model.train_loader)

    # Training loss dispatch
    if model.encoder.bool_config.variational

        @reset model.log_prior.normalize = true

        # Cyclic beta annealing (Fu et al., 2019)
        max_kl_weight = parse(Float32, retrieve(conf, "VARIATIONAL", "beta"))
        beta_num_cycles = parse(Int, retrieve(conf, "VARIATIONAL", "num_cycles"))
        beta_cycle_length = parse(Int, retrieve(conf, "VARIATIONAL", "cycle_length"))
        annealing_fraction = parse(Float32, retrieve(conf, "VARIATIONAL", "annealing_fraction"))
        beta = [max_kl_weight]

        if beta_num_cycles > 0
            annealing_steps = floor(Int, beta_cycle_length * annealing_fraction)
            beta = Vector{Float32}(undef, num_param_updates + 1)
            for step in 1:(num_param_updates + 1)
                current_cycle = fld(step - 1, beta_cycle_length + 1)
                if current_cycle >= beta_num_cycles
                    beta[step] = max_kl_weight
                else
                    cycle_position = (step - 1) % (beta_cycle_length + 1)
                    if cycle_position <= annealing_steps
                        beta[step] = max_kl_weight * (cycle_position / annealing_steps)
                    else
                        beta[step] = max_kl_weight
                    end
                end
            end
        end

        static_loss = VariationalLoss(model, beta)
        @reset model.train_step = maybe_compile(
            static_loss, MLIR, opt_state, ps, st_kan, st_lux, x, st_rng,
        )
        println("Posterior sampler: Variational")

    elseif model.N_t > 1

        Q, S = model.prior.q_size, model.batch_size
        P = model.prior.bool_config.mixture_model ? 1 : model.prior.p_size

        @reset model.train_step = begin
            thermo_model = model
            if !model.lkhood.SEQ && !model.lkhood.CNN
                for i in 1:model.lkhood.generator.depth
                    @reset thermo_model.lkhood.generator.Φ_fcns[i].basis_function.S = S * (model.N_t + 1)
                end
            end
            @reset thermo_model.lkhood.generator.s_size = S * (model.N_t + 1)
            static_loss = ThermoLoss(thermo_model)
            maybe_compile(static_loss, MLIR, opt_state, ps, st_kan, st_lux, x, st_rng)
        end
        println("Posterior sampler: Thermo $(uppercase(model.sampler_type))")

    elseif model.sampler_type != "importance" || model.prior.bool_config.ula

        static_loss = LangevinLoss(model)
        @reset model.train_step = maybe_compile(
            static_loss, MLIR, opt_state, ps, st_kan, st_lux, x, st_rng,
        )
        println("Posterior sampler: MLE $(uppercase(model.sampler_type))")

    else

        static_loss = ImportanceLoss(model)
        @reset model.train_step = maybe_compile(
            static_loss, MLIR, opt_state, ps, st_kan, st_lux, x, st_rng,
        )
        println("Posterior sampler: MLE IS")
    end

    return model, st_lux, st_rng
end

function prep_model(
        model::KAEM{T},
        x::AbstractArray{T},
        optimizer;
        rng::AbstractRNG = Random.MersenneTwister(1),
        MLIR::Bool = true,
    ) where {T <: Float32}
    ps = Lux.initialparameters(rng, model)
    st_kan, st_lux = Lux.initialstates(rng, model)
    ps, st_kan, st_lux =
        ps |> ComponentArray |> Lux.f32 |> pu, st_kan |> Lux.f32 |> pu, st_lux |> Lux.f32 |> pu
    opt_state = Optimisers.setup(optimizer.rule(), ps)
    model, st_lux, st_rng = setup_training(
        opt_state,
        ps,
        st_kan,
        st_lux,
        model::KAEM{T},
        x;
        rng = rng,
        MLIR = MLIR
    )
    return model, opt_state, ps, st_kan, st_lux, st_rng
end

end
