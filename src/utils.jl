module Utils

export pu,
    xdev,
    symbol_map,
    activation_mapping,
    AbstractActivation,
    AbstractBasis,
    AbstractPrior,
    AbstractLogPrior,
    AbstractQuadrature,
    AbstractBoolConfig,
    AbstractResampler

using Lux, LinearAlgebra, Statistics, Random, Accessors, CUDA, LuxCUDA, NNlib, Reactant
using MLDataDevices: reactant_device
const pu = CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false")) ? gpu_device() : cpu_device()

Reactant.set_default_backend("cpu")
if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    CUDA.allowscalar(false)
    Reactant.set_default_backend("gpu")
    pu = gpu_device()
end

const xdev = reactant_device()

# Num layers must be flexible, yet static, so this is used to index into params/state
const symbol_map = (:a, :b, :c, :d, :e, :f, :g, :h, :i)

abstract type AbstractActivation end

struct ReluActivation <: AbstractActivation end
function (::ReluActivation)(x)
    return NNlib.relu(x)
end

struct LeakyReluActivation <: AbstractActivation end
function (::LeakyReluActivation)(x)
    return NNlib.leakyrelu(x)
end

struct TanhActivation <: AbstractActivation end
function (::TanhActivation)(x)
    return NNlib.tanh_fast(x)
end

struct SigmoidActivation <: AbstractActivation end
function (::SigmoidActivation)(x)
    return NNlib.sigmoid_fast(x)
end

struct SwishActivation <: AbstractActivation end
function (::SwishActivation)(x)
    return NNlib.hardswish(x)
end

struct GeluActivation <: AbstractActivation end
function (::GeluActivation)(x)
    return NNlib.gelu(x)
end

struct SeluActivation <: AbstractActivation end
function (::SeluActivation)(x)
    return NNlib.selu(x)
end

struct SiluActivation <: AbstractActivation end
function (::SiluActivation)(x)
    return x .* NNlib.sigmoid_fast(x)
end

struct EluActivation <: AbstractActivation end
function (::EluActivation)(x)
    return NNlib.elu(x)
end

struct CeluActivation <: AbstractActivation end
function (::CeluActivation)(x)
    return NNlib.celu(x)
end

struct NoneActivation <: AbstractActivation end
function (::NoneActivation)(x)
    return x .* 0.0f0
end

struct IdentityActivation <: AbstractActivation end
function (::IdentityActivation)(x)
    return x
end

struct SeqActivation <: AbstractActivation end
function (::SeqActivation)(x)
    return softmax(x; dims = 1)
end

const activation_mapping::Dict{String, AbstractActivation} = Dict(
    "relu" => ReluActivation(),
    "leakyrelu" => LeakyReluActivation(),
    "tanh" => TanhActivation(),
    "sigmoid" => SigmoidActivation(),
    "swish" => SwishActivation(),
    "gelu" => GeluActivation(),
    "selu" => SeluActivation(),
    "silu" => SiluActivation(),
    "elu" => EluActivation(),
    "celu" => CeluActivation(),
    "none" => NoneActivation(),
    "identity" => IdentityActivation(),
    "sequence" => SeqActivation(),
)

abstract type AbstractBasis end

abstract type AbstractPrior end

abstract type AbstractLogPrior end

abstract type AbstractQuadrature end

abstract type AbstractBoolConfig end

abstract type AbstractResampler end

end
