"""Warning: this script will not carry over optimizer state 
or updated seed - only the current model and parameters/st_luxate."""

using JLD2, Lux, ComponentArrays, ConfParser, Random

# EDIT:
dataset = "CIFAR10"
file_loc = "logs/Vanilla/n_z=100/ULA/cnn=true/$(dataset)_1/"
ckpt = 10

conf = Dict(
    "MNIST" => ConfParse("config/nist_config.ini"),
    "FMNIST" => ConfParse("config/nist_config.ini"),
    "CIFAR10" => ConfParse("config/cifar_config.ini"),
    "SVHN" => ConfParse("config/svhn_config.ini"),
    "CELEBA" => ConfParse("config/celeba_config.ini"),
    "PTB" => ConfParse("config/text_config.ini"),
    "SMS_SPAM" => ConfParse("config/text_config.ini"),
    "DARCY_FLOW" => ConfParse("config/darcy_flow_config.ini"),
)[dataset]
parse_conf!(conf)

ENV["DEVICE"] = retrieve(conf, "TRAINING", "device")
ENV["PERCEPTUAL"] = retrieve(conf, "TRAINING", "use_perceptual_loss")

# EDIT:
commit!(conf, "POST_LANGEVIN", "sampler", "pcnl")
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")

include("src/utils.jl")
include("src/pipeline/trainer.jl")
using .Utils: pu
using .trainer
using .trainer.KAEM_model: load_params

rng = Random.MersenneTwister(1)
t = init_trainer(rng, conf, dataset)

saved_data = load(file_loc * "ckpt_epoch_$ckpt.jld2")
t.ps = load_params(saved_data) |> pu
t.st_kan = saved_data["kan_state"] |> pu
t.st_lux = saved_data["lux_state"] |> pu
t.opt_state = saved_data["opt_state"] |> pu

train!(t)
