module ModelSetup

export prep_model, ULAPriorSampler

using ConfParser, Lux, Accessors, ComponentArrays, Random, Reactant, Enzyme, Optimisers

using ..Utils
using ..KAEM_model
using ..KAEM_model.EBM_Model: prior_sampler_kind
using ..KAEM_model.LogPriorFCNs
using ..KAEM_model.InverseTransformSampling
using ..KAEM_model.PopulationXchange

include("train_steps/langevin_mle.jl")
include("train_steps/importance_sampling.jl")
include("train_steps/thermodynamic.jl")
include("train_steps/variational.jl")
include("posterior_sampling/langevin.jl")
include("rng.jl")
using .ImportanceSampling
using .LangevinMLE
using .ThermodynamicIntegration
using .VariationalTraining
using .LangevinSampling
using .HLOrng

function maybe_compile(loss, MLIR, opt_state, ps, st_kan, st_lux, x, st_rng)
    return MLIR ? Reactant.@compile(loss(opt_state, ps, st_kan, st_lux, x, 1, st_rng)) : loss
end

struct ULAPriorSampler{S, X}
    ula::S
    x_proxy::X
end
(s::ULAPriorSampler)(ps, st_kan, st_lux, st_rng) =
    s.ula(ps, st_kan, Lux.trainmode(st_lux), s.x_proxy, st_rng)

### Prior sampler dispatch
function setup_prior_sampler(::PriorULA, model, conf, x)
    N = parse(Int, retrieve(conf, "PRIOR_LANGEVIN", "iters"))
    η = parse(Float32, retrieve(conf, "PRIOR_LANGEVIN", "step_size"))
    ula = initialize_ULA_sampler(model; η = η, N = N, prior_sampling_bool = true)
    @reset model.log_prior = LogPriorULA(model.ε)
    @reset model.sample_prior = ULAPriorSampler(ula, x)
    println("Prior sampler: ULA")
    return model
end

function setup_prior_sampler(::PriorMixITS, model, conf, x)
    @reset model.sample_prior = MixITSSampler(model.prior)
    @reset model.log_prior =
        LogPriorMix(model.ε, !model.prior.bool_config.contrastive_div)
    println("Prior sampler: Mix ITS")
    return model
end

function setup_prior_sampler(::PriorUnivITS, model, conf, x)
    @reset model.sample_prior = UnivITSSampler(model.prior)
    @reset model.log_prior =
        LogPriorUnivariate(model.ε, !model.prior.bool_config.contrastive_div)
    println("Prior sampler: Univar ITS")
    return model
end

### Posterior sampler dispatch
function setup_posterior_sampler(::PosteriorPCNL, model, conf, num_steps, η, exchange_type)
    δ = parse(Float32, retrieve(conf, "POST_LANGEVIN", "pcnl_delta"))
    @reset model.posterior_sampler = initialize_pCNL_sampler(
        model; δ = δ, N = num_steps, exchange_type = exchange_type,
    )
    return model
end

function setup_posterior_sampler(::PosteriorULA, model, conf, num_steps, η, exchange_type)
    @reset model.posterior_sampler = initialize_ULA_sampler(
        model; η = η, N = num_steps, exchange_type = exchange_type,
    )
    return model
end

function setup_posterior_sampler(::PosteriorImportance, model, conf, num_steps, η, exchange_type)
    @reset model.posterior_sampler = initialize_pCNL_sampler(model; N = num_steps)
    return model
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

    model = setup_prior_sampler(prior_sampler_kind(model.prior), model, conf, x)

    exchange_type = retrieve(conf, "THERMODYNAMIC_INTEGRATION", "exchange_type")
    model = setup_posterior_sampler(
        posterior_sampler_kind(model.sampler_type),
        model, conf, num_steps, η, exchange_type,
    )

    st_rng = seed_rand(model; rng = rng)

    # Forward pass to init st_lux state before compilation
    _, st_ebm, st_gen = Reactant.@jit model(ps, st_kan, Lux.trainmode(st_lux), st_rng)
    @reset st_lux.ebm = st_ebm
    @reset st_lux.gen = st_gen

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
        lr_ebm::T,
    ) where {T <: Float32}
    ps = Lux.initialparameters(rng, model)
    st_kan, st_lux = Lux.initialstates(rng, model)
    ps, st_kan, st_lux =
        ps |> ComponentArray |> Lux.f32 |> pu, st_kan |> Lux.f32 |> pu, st_lux |> Lux.f32 |> pu
    opt_state = Optimisers.setup(optimizer.rule(), ps)
    # Two-rate optimization for CD (Pang et al., 2020): independent learning rate
    # for the EBM subtree. Matches lr_ebm == lr_gen ⇒ effective single-rate.
    Optimisers.adjust!(opt_state.ebm, lr_ebm)
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
