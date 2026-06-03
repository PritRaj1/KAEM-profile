ENV["DEVICE"] = "cpu"
using Random, Lux, ConfParser, ComponentArrays, Optimisers

include("src/utils.jl")
using .Utils

include("src/KAEM/KAEM.jl")
using .KAEM_model

include("src/KAEM/model_setup.jl")
using .ModelSetup

include("src/pipeline/optimizer.jl")
using .optimization

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")
commit!(conf, "VARIATIONAL", "use_variational", "false")

optimizer = create_opt(conf)
lr_ebm = parse(Float32, retrieve(conf, "OPTIMIZER", "ebm_learning_rate"))
rng = Random.MersenneTwister(1)

dataset = randn(rng, Float32, 32, 32, 1, 500)
model = init_KAEM(dataset, conf, (32, 32, 1))
x_test = first(model.train_loader) |> pu
model, opt_state, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; rng = rng, MLIR = false, lr_ebm = lr_ebm)

println("=== opt_state structure ===")
println("typeof: ", typeof(opt_state))
println("fields: ", propertynames(opt_state))
println("opt_state.gen.rule.eta: ", opt_state.gen.rule.eta)
println("opt_state.ebm.rule.eta: ", opt_state.ebm.rule.eta)
