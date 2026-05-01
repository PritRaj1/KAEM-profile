using Test, Random, LinearAlgebra, Statistics, Lux, ConfParser, ComponentArrays, Reactant

ENV["DEVICE"] = "gpu"

include("../src/utils.jl")
using .Utils

include("../src/KAEM/KAEM.jl")
using .KAEM_model

include("../src/KAEM/model_setup.jl")
using .ModelSetup

include("../src/pipeline/optimizer.jl")
using .optimization

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
b_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
p_size = first(parse.(Int, retrieve(conf, "EbmModel", "layer_widths")))
q_size = last(parse.(Int, retrieve(conf, "EbmModel", "layer_widths")))

rng = Random.MersenneTwister(1)
dataset = randn(rng, Float32, 32, 32, 1, b_size * 10)
model = init_KAEM(dataset, conf, (32, 32, 1))
x_test = first(model.train_loader) |> pu
optimizer = create_opt(conf)
model, _, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; MLIR = false)

function test_shapes()
    @test model.prior.p_size == p_size
    return @test model.prior.q_size == q_size
end

function test_uniform_prior()
    commit!(conf, "EbmModel", "π_0", "uniform")

    compiled_sample = Reactant.@compile model.sample_prior(ps, st_kan, st_lux, st_rng)
    z_test = first(compiled_sample(ps, st_kan, st_lux, st_rng))

    if model.prior.bool_config.mixture_model || model.prior.bool_config.ula
        @test size(z_test) == (q_size, 1, b_size)
    else
        @test size(z_test) == (q_size, p_size, b_size)
    end

    compiled_log_prior = Reactant.@compile model.log_prior(
        z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad; st_rng = st_rng,
    )
    log_p = first(
        compiled_log_prior(
            z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad; st_rng = st_rng,
        )
    )

    @test !any(isnan, Array(z_test))
    @test size(log_p) == (b_size,)
    return @test !any(isnan, Array(log_p))
end

function test_gaussian_prior()
    commit!(conf, "EbmModel", "π_0", "gaussian")
    compiled_sample = Reactant.@compile model.sample_prior(ps, st_kan, st_lux, st_rng)
    z_test = first(compiled_sample(ps, st_kan, st_lux, st_rng))

    if model.prior.bool_config.mixture_model || model.prior.bool_config.ula
        @test all(size(z_test) .== (q_size, 1, b_size))
    else
        @test all(size(z_test) .== (q_size, p_size, b_size))
    end

    compiled_log_prior = Reactant.@compile model.log_prior(
        z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad; st_rng = st_rng,
    )
    log_p = first(
        compiled_log_prior(
            z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad; st_rng = st_rng,
        )
    )

    @test !any(isnan, Array(z_test))
    @test size(log_p) == (b_size,)
    @test !any(isnan, Array(log_p))

    z_array = Array(z_test)
    z_flat = reshape(z_array, :)

    @test abs(mean(z_flat)) < 2.0f0  # Roughly centered
    @test std(z_flat) > 1.0f-2  # Not degenerate
    @test all(Array(log_p) .> -1.0f6)  # Not extreme
    return @test all(Array(log_p) .< 1.0f6)
end

function test_lognormal_prior()
    commit!(conf, "EbmModel", "π_0", "lognormal")
    compiled_sample = Reactant.@compile model.sample_prior(ps, st_kan, st_lux, st_rng)
    z_test = first(compiled_sample(ps, st_kan, st_lux, st_rng))

    if model.prior.bool_config.mixture_model || model.prior.bool_config.ula
        @test all(size(z_test) .== (q_size, 1, b_size))
    else
        @test all(size(z_test) .== (q_size, p_size, b_size))
    end

    compiled_log_prior = Reactant.@compile model.log_prior(
        z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad; st_rng = st_rng,
    )
    log_p = first(
        compiled_log_prior(
            z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad; st_rng = st_rng,
        )
    )

    @test !any(isnan, Array(z_test))
    @test size(log_p) == (b_size,)
    return @test !any(isnan, Array(log_p))
end

function test_learnable_gaussian_prior()
    commit!(conf, "EbmModel", "π_0", "learnable_gaussian")
    compiled_sample = Reactant.@compile model.sample_prior(ps, st_kan, st_lux, st_rng)
    z_test = first(compiled_sample(ps, st_kan, st_lux, st_rng))

    if model.prior.bool_config.mixture_model || model.prior.bool_config.ula
        @test all(size(z_test) .== (q_size, 1, b_size))
    else
        @test all(size(z_test) .== (q_size, p_size, b_size))
    end

    compiled_log_prior = Reactant.@compile model.log_prior(
        z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad; st_rng = st_rng,
    )
    log_p = first(
        compiled_log_prior(
            z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad; st_rng = st_rng,
        )
    )

    @test !any(isnan, Array(z_test))
    @test size(log_p) == (b_size,)
    return @test !any(isnan, Array(log_p))
end

function test_ebm_prior()
    commit!(conf, "EbmModel", "π_0", "ebm")
    compiled_sample = Reactant.@compile model.sample_prior(ps, st_kan, st_lux, st_rng)
    z_test = first(compiled_sample(ps, st_kan, st_lux, st_rng))

    if model.prior.bool_config.mixture_model || model.prior.bool_config.ula
        @test all(size(z_test) .== (q_size, 1, b_size))
    else
        @test all(size(z_test) .== (q_size, p_size, b_size))
    end

    compiled_log_prior = Reactant.@compile model.log_prior(
        z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad; st_rng = st_rng,
    )
    log_p = first(
        compiled_log_prior(
            z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad; st_rng = st_rng,
        )
    )

    @test !any(isnan, Array(z_test))
    @test size(log_p) == (b_size,)
    return @test !any(isnan, Array(log_p))
end

function test_kl_gaussian_prior()
    commit!(conf, "EbmModel", "π_0", "kl_gaussian")
    commit!(conf, "PCA", "use_pca", "true")
    commit!(conf, "PCA", "pca_components", string(p_size))

    kl_dataset = randn(rng, Float32, 32, 32, 1, b_size * 10)
    kl_model = init_KAEM(kl_dataset, conf, (32, 32, 1))
    x_kl = first(kl_model.train_loader) |> pu
    kl_opt = create_opt(conf)
    kl_model, _, kl_ps, kl_st_kan, kl_st_lux, kl_st_rng =
        prep_model(kl_model, x_kl, kl_opt; MLIR = false)

    @test kl_model.prior.prior_type == "kl_gaussian"
    σ_host = Array(kl_model.prior.π_pdf.σ)
    @test length(σ_host) == p_size
    @test all(σ_host .> 0)

    compiled_sample = Reactant.@compile kl_model.sample_prior(
        kl_ps, kl_st_kan, kl_st_lux, kl_st_rng,
    )
    z_test = first(
        compiled_sample(kl_ps, kl_st_kan, kl_st_lux, kl_st_rng),
    )

    if kl_model.prior.bool_config.mixture_model || kl_model.prior.bool_config.ula
        @test size(z_test) == (q_size, 1, b_size)
    else
        @test size(z_test) == (q_size, p_size, b_size)
    end
    @test !any(isnan, Array(z_test))

    compiled_log_prior = Reactant.@compile kl_model.log_prior(
        z_test,
        kl_model.prior,
        kl_ps.ebm,
        kl_st_kan.ebm,
        kl_st_lux.ebm,
        kl_st_kan.quad;
        st_rng = kl_st_rng,
    )
    log_p = first(
        compiled_log_prior(
            z_test,
            kl_model.prior,
            kl_ps.ebm,
            kl_st_kan.ebm,
            kl_st_lux.ebm,
            kl_st_kan.quad;
            st_rng = kl_st_rng,
        ),
    )

    @test size(log_p) == (b_size,)
    return @test !any(isnan, Array(log_p))
end

@testset "Mixture Prior Tests" begin
    test_shapes()
    test_uniform_prior()
    test_gaussian_prior()
    test_lognormal_prior()
    test_learnable_gaussian_prior()
    test_ebm_prior()
    test_kl_gaussian_prior()
end
