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
    AbstractBoolConfig

using Lux, LinearAlgebra, Statistics, Random, Accessors, BFloat16s, CUDA, LuxCUDA, NNlib, Reactant

const pu =
    (CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))) ? gpu_device() : cpu_device()

if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    CUDA.allowscalar(false)
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

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
(::ReluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant} =
    NNlib.relu(x)

struct LeakyReluActivation <: AbstractActivation end
(::LeakyReluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant} =
    NNlib.leakyrelu(x)

struct TanhActivation <: AbstractActivation end
(::TanhActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant} =
    NNlib.tanh_fast(x)

struct SigmoidActivation <: AbstractActivation end
(::SigmoidActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant} =
    NNlib.sigmoid_fast(x)

struct SwishActivation <: AbstractActivation end
(::SwishActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant} =
    NNlib.hardswish(x)

struct GeluActivation <: AbstractActivation end
(::GeluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant} =
    NNlib.gelu(x)

struct SeluActivation <: AbstractActivation end
(::SeluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant} =
    NNlib.selu(x)

struct SiluActivation <: AbstractActivation end
(::SiluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant} =
    x .* NNlib.sigmoid_fast(x)

struct EluActivation <: AbstractActivation end
(::EluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant} =
    NNlib.elu(x)

struct CeluActivation <: AbstractActivation end
(::CeluActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant} =
    NNlib.celu(x)

struct NoneActivation <: AbstractActivation end
(::NoneActivation)(x::AbstractArray{T})::AbstractArray{T} where {T <: half_quant} =
    x .* zero(T)

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
)

abstract type AbstractBasis end

abstract type AbstractPrior end

abstract type AbstractLogPrior end

abstract type AbstractQuadrature end

abstract type AbstractBoolConfig end

end
