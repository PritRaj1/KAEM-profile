module Utils

export pu,
    half_quant,
    full_quant,
    hq,
    fq,
    symbol_map,
    activation_mapping,
    AbstractActivation,
    AbstractBasis,
    AbstractPrior,
    AbstractLogPrior,
    AbstractQuadrature,
    AbstractBoolConfig,
    AbstractResampler

using Lux, LinearAlgebra, Statistics, Random, Accessors, BFloat16s, CUDA, LuxCUDA, NNlib, Reactant
using MLDataDevices: reactant_device

# const pu =
#     (CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))) ? gpu_device() : cpu_device()

Reactant.set_default_backend("cpu")
if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    CUDA.allowscalar(true)
    Reactant.set_default_backend("gpu")
end

const pu = reactant_device()

# # Mixed precision - sometimes unstable, use FP16 when Tensor Cores are available
const QUANT_MAP =
    Dict("BF16" => BFloat16, "FP16" => Float16, "FP32" => Float32, "FP64" => Float64)

const LUX_QUANT_MAP =
    Dict("BF16" => Lux.bf16, "FP16" => Lux.f16, "FP32" => Lux.f32, "FP64" => Lux.f64)

const half_quant = get(QUANT_MAP, uppercase(get(ENV, "HALF_QUANT", "FP32")), Float32)
const full_quant = get(QUANT_MAP, uppercase(get(ENV, "FULL_QUANT", "FP32")), Float32)
const hq = get(LUX_QUANT_MAP, uppercase(get(ENV, "HALF_QUANT", "FP32")), Lux.f32)
const fq = get(LUX_QUANT_MAP, uppercase(get(ENV, "FULL_QUANT", "FP32")), Lux.f32)

# Num layers must be flexible, yet static, so this is used to index into params/state
const symbol_map = (:a, :b, :c, :d, :e, :f, :g, :h, :i)

abstract type AbstractActivation end

struct ReluActivation <: AbstractActivation end
function (::ReluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant}
    return NNlib.relu(x)
end

struct LeakyReluActivation <: AbstractActivation end
function (::LeakyReluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant}
    return NNlib.leakyrelu(x)
end

struct TanhActivation <: AbstractActivation end
function (::TanhActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant}
    return NNlib.tanh_fast(x)
end

struct SigmoidActivation <: AbstractActivation end
function (::SigmoidActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant}
    return NNlib.sigmoid_fast(x)
end

struct SwishActivation <: AbstractActivation end
function (::SwishActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant}
    return NNlib.hardswish(x)
end

struct GeluActivation <: AbstractActivation end
function (::GeluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant}
    return NNlib.gelu(x)
end

struct SeluActivation <: AbstractActivation end
function (::SeluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant}
    return NNlib.selu(x)
end

struct SiluActivation <: AbstractActivation end
function (::SiluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant}
    return x .* NNlib.sigmoid_fast(x)
end

struct EluActivation <: AbstractActivation end
function (::EluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant}
    return NNlib.elu(x)
end

struct CeluActivation <: AbstractActivation end
function (::CeluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant}
    return NNlib.celu(x)
end

struct NoneActivation <: AbstractActivation end
function (::NoneActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant}
    return x .* zero(T)
end

struct IdentityActivation <: AbstractActivation end
function (::IdentityActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant}
    return x
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
)

abstract type AbstractBasis end

abstract type AbstractPrior end

abstract type AbstractLogPrior end

abstract type AbstractQuadrature end

abstract type AbstractBoolConfig end

abstract type AbstractResampler end

end
