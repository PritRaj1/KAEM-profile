module Losses

export IS_loss, MALA_loss

using ..Utils

using CUDA, Statistics, Lux
using NNlib: conv, batched_mul

perceptual_loss = parse(Bool, get(ENV, "PERCEPTUAL", "false"))
feature_extractor = nothing
style_lyrs = [2, 5, 9, 12]
content_lyrs = [9]
if perceptual_loss
    using Metalhead: VGG
    feature_extractor = VGG(16; pretrain = true).layers[1][1:12] |> Lux.f32 |> pu # Conv layers only, (rest is classifier)
end


## Fcns for model with Importance Sampling ##
function cross_entropy_IS(
        x::AbstractArray{T, 3},
        x̂::AbstractArray{T, 4},
        ε::T,
        scale::T,
    )::AbstractArray{T, 2} where {T <: Float32}
    D, L, S, B = size(x̂)
    ll =
        dropdims(sum(log.(x̂ .+ ε) .* reshape(x, D, L, 1, B), dims = (1, 2)), dims = (1, 2))
    return ll' ./ T(D) ./ scale
end

function l2_IS(
        x::AbstractArray{T, 4},
        x̂::AbstractArray{T, 5},
        ε::T,
        scale::T,
    )::AbstractArray{T, 2} where {T <: Float32}
    W, H, C, S, B = size(x̂)
    ll =
        -dropdims(
        sum((reshape(x, W, H, C, 1, B) .- x̂) .^ 2, dims = (1, 2, 3)),
        dims = (1, 2, 3),
    )
    return ll' ./ scale
end

function l2_IS_PCA(
        x::AbstractArray{T, 2},
        x̂::AbstractArray{T, 3},
        ε::T,
        scale::T,
    )::AbstractArray{T, 2} where {T <: Float32}
    D, S, B = size(x̂)
    ll = -dropdims(sum((reshape(x, D, 1, B) .- x̂) .^ 2, dims = 1), dims = 1)
    return ll' ./ scale
end

function IS_loss(
        x::AbstractArray{T},
        x̂::AbstractArray{T},
        ε::T,
        scale::T,
        B::Int,
        S::Int,
        SEQ::Bool,
    )::AbstractArray{T, 2} where {T <: Float32}
    loss_fcn = (SEQ ? cross_entropy_IS : (ndims(x) == 2 ? l2_IS_PCA : l2_IS))
    return loss_fcn(x, x̂, ε, scale)
end

## Fcns for model with Langevin methods ##
function cross_entropy_MALA(
        x::AbstractArray{T, 3},
        x̂::AbstractArray{T, 3},
        ε::T,
        scale::T,
    )::AbstractArray{T, 1} where {T <: Float32}
    ll = dropdims(sum(log.(x̂ .+ ε) .* x, dims = (1, 2)), dims = (1, 2))
    return ll ./ T(size(x, 1)) ./ scale
end

function l2_PCA(
        x::AbstractArray{T, 2},
        x̂::AbstractArray{T, 2},
        ε::T,
        scale::T,
        perceptual_scale::T,
    )::AbstractArray{T, 1} where {T <: Float32}
    ll = -dropdims(sum((x .- x̂) .^ 2, dims = 1), dims = 1)
    return ll ./ scale
end

function l2_MALA(
        x::AbstractArray{T, 4},
        x̂::AbstractArray{T, 4},
        ε::T,
        scale::T,
        perceptual_scale::T,
    )::AbstractArray{T, 1} where {T <: Float32}
    ll = -dropdims(sum((x .- x̂) .^ 2, dims = (1, 2, 3)), dims = (1, 2, 3))
    return ll ./ scale
end

# Gaussian SSIM kernel
const SSIM_KERNEL =
    [
    0.00102838008447911,
    0.007598758135239185,
    0.03600077212843083,
    0.10936068950970002,
    0.2130055377112537,
    0.26601172486179436,
    0.2130055377112537,
    0.10936068950970002,
    0.03600077212843083,
    0.007598758135239185,
    0.00102838008447911,
] .|> Float32
const C₁ = 0.01^2 |> Float32
const C₂ = 0.03^2 |> Float32
const kernel = repeat(reshape(SSIM_KERNEL * SSIM_KERNEL', 11, 11, 1, 1), 1, 1, 3, 1) |> pu

function ssim_MALA(
        x::AbstractArray{T, 4},
        x̂::AbstractArray{T, 4},
        ε::T,
        scale::T,
        perceptual_scale::T,
    )::AbstractArray{T, 1} where {T <: Float32}
    μx = conv(x, kernel)
    μy = conv(x̂, kernel)
    μx² = μx .^ 2
    μy² = μy .^ 2
    μxy = μx .* μy
    σx² = conv(x .^ 2, kernel) .- μx²
    σy² = conv(x̂ .^ 2, kernel) .- μy²
    σxy = conv(x .* x̂, kernel) .- μxy

    ssim_map = @. (2μxy + C₁) * (2σxy + C₂) / ((μx² + μy² + C₁) * (σx² + σy² + C₂))
    return dropdims(mean(ssim_map, dims = (1, 2, 3)), dims = (1, 2, 3)) ./ perceptual_scale
end

function gramm_loss(
        x::AbstractArray{T, 4},
        x̂::AbstractArray{T, 4},
        scale::T,
    )::AbstractArray{T, 1} where {T <: Float32}
    H, W, C, B = size(x)
    real = reshape(x, H * W, C, B)
    fake = reshape(x̂, H * W, C, B)
    real_perm = permutedims(real, (2, 1, 3))
    fake_perm = permutedims(fake, (2, 1, 3))
    G_real = batched_mul(real_perm, real)
    G_fake = batched_mul(fake_perm, fake)
    return -dropdims(sum((G_real .- G_fake) .^ 2, dims = (1, 2)), dims = (1, 2)) ./ scale
end

function feature_loss(
        x::AbstractArray{T, 4},
        x̂::AbstractArray{T, 4},
        ε::T,
        scale::T,
        perceptual_scale::T,
    )::AbstractArray{T, 1} where {T <: Float32}
    loss = l2_MALA(x, x̂, ε, scale, perceptual_scale)
    real_features, fake_features = x, x̂
    for (idx, layer) in enumerate(feature_extractor)
        scale_at_lyr = prod(size(real_features)[1:3]) * scale
        real_features, fake_features = layer(real_features), layer(fake_features)
        loss =
            (idx in style_lyrs) ?
            perceptual_scale .* gramm_loss(real_features, fake_features, scale_at_lyr) +
            loss : loss

        loss =
            (idx in content_lyrs) ?
            perceptual_scale .*
            l2_MALA(real_features, fake_features, ε, scale_at_lyr, perceptual_scale) +
            loss : loss
    end
    return loss
end

function MALA_loss(
        x::AbstractArray{T},
        x̂::AbstractArray{T},
        ε::T,
        scale::T,
        B::Int,
        SEQ::Bool,
        perceptual_scale::T,
    )::AbstractArray{T, 1} where {T <: Float32}
    loss_fcn = (
        SEQ ? cross_entropy_MALA :
            (ndims(x) == 2 ? l2_PCA : (perceptual_loss ? feature_loss : l2_MALA))
    )
    return loss_fcn(x, x̂, ε, scale, perceptual_scale)
end

end
