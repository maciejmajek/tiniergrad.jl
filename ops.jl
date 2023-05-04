import Base: *
import LinearAlgebra: mul!
using LinearAlgebra

FA = Union{Float64, AbstractArray{Float64}}
A = AbstractArray{Float64}

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x::Float64, y::Float64) = return x .+ y
backward(::BroadcastedOperator{typeof(+)}, x::Float64, y::Float64, g::Float64) = tuple(g, g)

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x::Float64, y::Float64) = return x .- y
backward(::BroadcastedOperator{typeof(-)}, x::Float64, y::Float64, g::Float64) = tuple(g, -g)

*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, AA::A, x::A) = return AA * x
backward(::BroadcastedOperator{typeof(mul!)}, AA::A, x::A, g::A) = tuple(g * x', AA' * g)

Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x::FA, y) = return x .* y
backward(node::BroadcastedOperator{typeof(*)}, x::FA, y::FA, g::FA) =
    let
        𝟏 = ones(length(node.output))
        Jx = diagm(y .* 𝟏)
        Jy = diagm(x .* 𝟏)
        tuple(Jx' * g, Jy' * g)
    end

import Base: ^
^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x::FA, n::FA) = return x^n
backward(::ScalarOperator{typeof(^)}, x::FA, n::FA, g::FA) =
    tuple(g * n * x^(n - 1), g * log(abs(x)) * x^n)

import Base: log
log(x::GraphNode) = ScalarOperator(log, x)
forward(::ScalarOperator{typeof(log)}, x::FA) = return log(x)
backward(::ScalarOperator{typeof(log)}, x::FA, g::FA) = return g / x

import Base: sum
sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x::FA) = return sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x::FA, g::FA) = return tuple(ones(size(x)) .* g)

import Base: log
Base.Broadcast.broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x::FA) = return log.(x)
backward(::BroadcastedOperator{typeof(log)}, x::FA, g::FA) =
    let
        𝟏 = ones(length(x))
        J = 𝟏' ./ x
        tuple(J' * g)
    end

Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x::FA, y::FA) = return x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x::FA, y::FA, g::FA) =
    let
        𝟏 = ones(length(node.output))
        Jx = diagm(𝟏 ./ y)
        Jy = (-x ./ y .^ 2)
        tuple(Jx' * g, Jy' * g)
    end

import Base: max
Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x::FA, y::FA) = return max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x::FA, y::FA, g::FA) =
    let
        Jx = diagm(isless.(y, x))
        Jy = diagm(isless.(x, y))
        tuple(Jx' * g, Jy' * g)
    end

sigmoid(x::GraphNode) = BroadcastedOperator(sigmoid, x)
forward(::BroadcastedOperator{typeof(sigmoid)}, x) = 1 ./ (1 .+ exp.(-x))
backward(node::BroadcastedOperator{typeof(sigmoid)}, x::A, g::A) =
    tuple(g .* node.output .* (1 .- node.output))


relu(x::GraphNode) = BroadcastedOperator(relu, x)
forward(::BroadcastedOperator{typeof(relu)}, x::A) = return max.(x, zero(x))
backward(node::BroadcastedOperator{typeof(relu)}, x::A, g::A) = return tuple(g .* (x .> 0))

cross_entropy_loss(y_hat::GraphNode, y::GraphNode) =
    BroadcastedOperator(cross_entropy_loss, y_hat, y)
forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat::A, y::A) =
    let
        y_hat = y_hat .- maximum(y_hat)
        y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
        loss = sum(log.(y_hat) .* y) * -1.0
        return loss
    end
backward(node::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat::A, y::A, g::FA) =
    let
        y_hat = y_hat .- maximum(y_hat)
        y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
        return tuple(g .* (y_hat - y))
    end

Base.Broadcast.broadcasted(^, x::GraphNode, y::GraphNode) = BroadcastedOperator(^, x, y)
forward(::BroadcastedOperator{typeof(^)}, x::FA, y::FA) = return x .^ y
backward(::BroadcastedOperator{typeof(^)}, x::FA, y::FA, g::FA) =
    let
        gx = g .* (y .* (x .^ (y .- 1)))
        gy = g .* (x .^ y) .* log.(x)
        return (gx, gy)
    end

softmax(x::GraphNode) = BroadcastedOperator(softmax, x)
forward(::BroadcastedOperator{typeof(softmax)}, x::A) = exp.(x) ./ sum(exp.(x))
backward(node::BroadcastedOperator{typeof(softmax)}, x::A, g::A) =
    let
        y = node.output
        J = diagm(y) .- y * y'
        tuple(J' * g)
    end

import NNlib


function conv(x::A, kernel::A; pad = 0, flipped = false)
    h, w, c = size(x)
    kh, kw, kc, kb = size(kernel)

    if ~flipped
        for i = 1:kc
            for j = 1:kb
                kernel[:, :, i, j] = rot180(kernel[:, :, i, j])
            end
        end
    end

    output_h = h + 2 * pad - (kh - 1)
    output_w = w + 2 * pad - (kw - 1)
    if pad != 0
        x1 = zeros(output_h + floor(Int, kh / 2) * 2, output_w + floor(Int, kw / 2) * 2, c)
        x1[1+pad:end-pad, 1+pad:end-pad, :] = x
        x = x1
    end

    output = zeros(output_h, output_w, kb)
    for channel = 1:kb
        for i = 1:output_h
            for j = 1:output_w
                output[i, j, channel] =
                    sum(kernel[:, :, :, channel] .* x[i:i+kh-1, j:j+kw-1, :])
            end
        end
    end
    return output
end

add_dim(x::Array) = reshape(x, (size(x)..., 1))
conv2d(x::GraphNode, kernel::GraphNode) = BroadcastedOperator(conv2d, x, kernel)
forward(::BroadcastedOperator{typeof(conv2d)}, x::A, kernel::A) =
    let
        input = @view x[:, :, :, 1:1]
        output = @view conv_op(input, kernel, flipped = true)[:, :, :, 1]
        return output
    end

backward(::BroadcastedOperator{typeof(conv2d)}, x::A, kernel::A, g::A) =
    let
        x = add_dim(x)
        if size(g)[end] != 1
            g = add_dim(g)
        end

        kernel_gradient = permutedims(
            conv_op(
                permutedims(x, (1, 2, 4, 3)),
                permutedims(g, (1, 2, 4, 3)),
                flipped = true,
            ),
            (1, 2, 4, 3),
        )
        input_gradient =
            conv_op(g, permutedims(kernel, (1, 2, 4, 3)), pad = 2, flipped = false)
        return tuple(input_gradient, kernel_gradient)
    end

import Base.reshape
reshape(x::GraphNode, new_size::GraphNode) = BroadcastedOperator(reshape, x, new_size)
forward(::BroadcastedOperator{typeof(reshape)}, x::A, new_size::Tuple{Int64}) = reshape(x, new_size)
backward(::BroadcastedOperator{typeof(reshape)}, x::A, new_size::Tuple{Int64}, g::A) =
    tuple(reshape(g, size(x)))

function flatten() end
flatten(x::GraphNode) = BroadcastedOperator(flatten, x)
forward(::BroadcastedOperator{typeof(flatten)}, x::A) = reshape(x, length(x))
backward(::BroadcastedOperator{typeof(flatten)}, x::A, g::A) = tuple(reshape(g, size(x)))

function dense() end
dense(x::GraphNode, w::GraphNode, b::GraphNode) = BroadcastedOperator(dense, x, w, b)
forward(::BroadcastedOperator{typeof(dense)}, x::A, w::A, b::A) = w * x .+ b
backward(::BroadcastedOperator{typeof(dense)}, x::A, w::A, b::A, g::FA) = tuple(w' * g, g * x', g)

function maxpool2d() end
maxpool2d(x::GraphNode) = BroadcastedOperator(maxpool2d, x)
forward(node::BroadcastedOperator{typeof(maxpool2d)}, x::A) =
    let
        h, w, c = size(x)
        output = zeros(h ÷ 2, w ÷ 2, c)
        indices = CartesianIndex{3}[]
        for i = 1:c
            for j = 1:h÷2
                for k = 1:w÷2
                    val, ids = findmax(@view x[2*j-1:2*j, 2*k-1:2*k, i])
                    output[j, k, i] = val

                    idx, idy = ids[1] + 2 * j - 1 - 1, ids[2] + 2 * k - 1 - 1
                    push!(indices, CartesianIndex(idx, idy, i))
                end
            end
        end
        node.cache = indices
        output
    end

backward(node::BroadcastedOperator{typeof(maxpool2d)}, x::A, g::FA) =
    let
        output = zeros(size(x))
        output[node.cache] = vcat(g...)
        tuple(output)
    end
