using Test, Random, Lux, ConfParser, ComponentArrays, JLD2, Reactant
using MLDataDevices: cpu_device

ENV["DEVICE"] = "gpu"

include("../src/utils.jl")
using .Utils

include("../src/KAEM/KAEM.jl")
using .KAEM_model

include("../src/KAEM/model_setup.jl")
using .ModelSetup

include("../src/KAEM/grid_updating.jl")
using .ModelGridUpdating

include("../src/pipeline/optimizer.jl")
using .optimization

include("../src/KAEM/rng.jl")
using .HLOrng: seed_rand

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")
commit!(conf, "VARIATIONAL", "use_variational", "false")

optimizer = create_opt(conf)
rng = Random.MersenneTwister(1)

dataset = randn(rng, Float32, 32, 32, 1, 500)
model = init_KAEM(dataset, conf, (32, 32, 1))
x_test = first(model.train_loader) |> pu
model, opt_state, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; rng = rng)

result = model.train_step(opt_state, ps, st_kan, st_lux, x_test, 1, st_rng)
ps = result[2]

function test_axes()
    tmpfile = tempname() * ".jld2"

    jldsave(
        tmpfile;
        params_data = Array(getdata(ps)),
        params_axes = getaxes(ps),
        kan_state = st_kan |> cpu_device(),
        lux_state = st_lux |> cpu_device(),
        opt_state = opt_state |> cpu_device(),
    )

    saved = load(tmpfile)
    ps_loaded = load_params(saved)

    @test length(ps_loaded) == length(ps)
    @test Array(ps_loaded) ≈ Array(ps)
    @test hasproperty(ps_loaded, :ebm)
    @test hasproperty(ps_loaded, :gen)
    @test hasproperty(ps_loaded, :enc)
    @test hasproperty(ps_loaded.ebm, :fcn)
    @test Array(ps_loaded.ebm.fcn.a.w_base) ≈ Array(ps.ebm.fcn.a.w_base)

    sk = saved["kan_state"]
    sl = saved["lux_state"]
    @test keys(sk) == keys(st_kan)
    @test keys(sl) == keys(st_lux)
    @test Array(sk.quad.nodes) ≈ Array(st_kan.quad.nodes)
    @test haskey(saved, "opt_state")
    return rm(tmpfile)
end

function test_generate()
    tmpfile = tempname() * ".jld2"

    st_rng_a = seed_rand(model; rng = Random.MersenneTwister(99))
    gen_compiled = Reactant.@compile model(ps, st_kan, Lux.testmode(st_lux), st_rng_a)

    st_rng_1 = seed_rand(model; rng = Random.MersenneTwister(99))
    gen_before = Array(gen_compiled(ps, st_kan, Lux.testmode(st_lux), st_rng_1)[1])

    jldsave(
        tmpfile;
        params_data = Array(getdata(ps)),
        params_axes = getaxes(ps),
        kan_state = st_kan |> cpu_device(),
        lux_state = st_lux |> cpu_device(),
    )

    saved = load(tmpfile)
    ps2 = load_params(saved) |> pu
    st_kan2 = saved["kan_state"] |> pu
    st_lux2 = saved["lux_state"] |> pu

    st_rng_2 = seed_rand(model; rng = Random.MersenneTwister(99))
    gen_after = Array(gen_compiled(ps2, st_kan2, Lux.testmode(st_lux2), st_rng_2)[1])

    @test gen_after ≈ gen_before
    return rm(tmpfile)
end

@testset "Save/Load Tests" begin
    test_save_load_with_axes()
    test_save_load_generate()
end
