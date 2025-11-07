module Utils

export pu,
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

pu = cpu_device()
Reactant.set_default_backend("cpu")
if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    CUDA.allowscalar(false)
    Reactant.set_default_backend("gpu")
    pu = gpu_device()
end

# const pu = reactant_device()

# Num layers must be flexible, yet static, so this is used to index into params/state
const symbol_map = (:a, :b, :c, :d, :e, :f, :g, :h, :i)

abstract type AbstractActivation end

struct ReluActivation <: AbstractActivation end
function (::ReluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: Float32}
    return NNlib.relu(x)
end

struct LeakyReluActivation <: AbstractActivation end
function (::LeakyReluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: Float32}
    return NNlib.leakyrelu(x)
end

struct TanhActivation <: AbstractActivation end
function (::TanhActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: Float32}
    return NNlib.tanh_fast(x)
end

struct SigmoidActivation <: AbstractActivation end
function (::SigmoidActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: Float32}
    return NNlib.sigmoid_fast(x)
end

struct SwishActivation <: AbstractActivation end
function (::SwishActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: Float32}
    return NNlib.hardswish(x)
end

struct GeluActivation <: AbstractActivation end
function (::GeluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: Float32}
    return NNlib.gelu(x)
end

struct SeluActivation <: AbstractActivation end
function (::SeluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: Float32}
    return NNlib.selu(x)
end

struct SiluActivation <: AbstractActivation end
function (::SiluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: Float32}
    return x .* NNlib.sigmoid_fast(x)
end

struct EluActivation <: AbstractActivation end
function (::EluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: Float32}
    return NNlib.elu(x)
end

struct CeluActivation <: AbstractActivation end
function (::CeluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: Float32}
    return NNlib.celu(x)
end

struct NoneActivation <: AbstractActivation end
function (::NoneActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: Float32}
    return x .* 0.0f0
end

struct IdentityActivation <: AbstractActivation end
function (::IdentityActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: Float32}
    return x
end

struct SeqActivation <: AbstractActivation end
function (::SeqActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: Float32}
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
