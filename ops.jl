import Base: *
import LinearAlgebra: mul!
using LinearAlgebra

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = return x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = return x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g, -g)

*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = return x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) =
    let
        𝟏 = ones(length(node.output))
        Jx = diagm(y .* 𝟏)
        Jy = diagm(x .* 𝟏)
        tuple(Jx' * g, Jy' * g)
    end

import Base: ^
^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x, n) = return x^n
backward(::ScalarOperator{typeof(^)}, x, n, g) =
    tuple(g * n * x^(n - 1), g * log(abs(x)) * x^n)

import Base: log
log(x::GraphNode) = ScalarOperator(log, x)
forward(::ScalarOperator{typeof(log)}, x) = return log(x)
backward(::ScalarOperator{typeof(log)}, x, g) = return g / x

import Base: sum
sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = return sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) = return tuple(ones(size(x)) .* g)

import Base: log
Base.Broadcast.broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x) = return log.(x)
backward(::BroadcastedOperator{typeof(log)}, x, g) =
    let
        𝟏 = ones(length(x))
        J = 𝟏' ./ x
        tuple(J' * g)
    end

Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = return x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, g) =
    let
        𝟏 = ones(length(node.output))
        Jx = diagm(𝟏 ./ y)
        Jy = (-x ./ y .^ 2)
        tuple(Jx' * g, Jy' * g)
    end

import Base: max
Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = return max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) =
    let
        Jx = diagm(isless.(y, x))
        Jy = diagm(isless.(x, y))
        tuple(Jx' * g, Jy' * g)
    end

sigmoid(x::GraphNode) = BroadcastedOperator(sigmoid, x)
forward(::BroadcastedOperator{typeof(sigmoid)}, x) = 1 ./ (1 .+ exp.(-x))
backward(node::BroadcastedOperator{typeof(sigmoid)}, x, g) =
    tuple(g .* node.output .* (1 .- node.output))


relu(x::GraphNode) = BroadcastedOperator(relu, x)
forward(::BroadcastedOperator{typeof(relu)}, x) = return max.(x, zero(x))
backward(node::BroadcastedOperator{typeof(relu)}, x, g) = return tuple(g .* (x .> 0))

cross_entropy_loss(y_hat::GraphNode, y::GraphNode) =
    BroadcastedOperator(cross_entropy_loss, y_hat, y)
forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y) =
    let
        y_hat = y_hat .- maximum(y_hat)
        y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
        loss = sum(log.(y_hat) .* y) * -1.0
        return loss
    end
backward(node::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y, g) =
    let
        y_hat = y_hat .- maximum(y_hat)
        y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
        return tuple(g .* (y_hat - y))
    end

Base.Broadcast.broadcasted(^, x::GraphNode, y::GraphNode) = BroadcastedOperator(^, x, y)
forward(::BroadcastedOperator{typeof(^)}, x, y) = return x .^ y
backward(::BroadcastedOperator{typeof(^)}, x, y, g) =
    let
        gx = g .* (y .* (x .^ (y .- 1)))
        gy = g .* (x .^ y) .* log.(x)
        return (gx, gy)
    end

softmax(x::GraphNode) = BroadcastedOperator(softmax, x)
forward(::BroadcastedOperator{typeof(softmax)}, x) = exp.(x) ./ sum(exp.(x))
backward(node::BroadcastedOperator{typeof(softmax)}, x, g) =
    let
        y = node.output
        J = diagm(y) .- y * y'
        tuple(J' * g)
    end

import NNlib


function conv(x, kernel; pad=0, flipped=false)
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
forward(::BroadcastedOperator{typeof(conv2d)}, x, kernel) =
    let
        input = @view x[:, :, :, 1:1]
        output = @view conv_op(input, kernel, flipped=true)[:, :, :, 1]
        return output
    end

backward(::BroadcastedOperator{typeof(conv2d)}, x, kernel, g) =
    let
        x = add_dim(x)
        if size(g)[end] != 1
            g = add_dim(g)
        end
 
        kernel_gradient = permutedims(conv_op(permutedims(x, (1, 2, 4, 3)), permutedims(g, (1, 2, 4, 3)), flipped=true), (1, 2, 4, 3))
        input_gradient = conv_op(g, permutedims(kernel, (1, 2, 4, 3)), pad=2, flipped=false)
        return tuple(input_gradient, kernel_gradient)
    end

import Base.reshape
reshape(x::GraphNode, new_size::GraphNode) = BroadcastedOperator(reshape, x, new_size)
forward(::BroadcastedOperator{typeof(reshape)}, x, new_size) = reshape(x, new_size)
backward(::BroadcastedOperator{typeof(reshape)}, x, new_size, g) = tuple(reshape(g, size(x)))

function flatten() end
flatten(x::GraphNode) = BroadcastedOperator(flatten, x)
forward(::BroadcastedOperator{typeof(flatten)}, x) = reshape(x, length(x))
backward(::BroadcastedOperator{typeof(flatten)}, x, g) = tuple(reshape(g, size(x)))

function dense() end
dense(x::GraphNode, w::GraphNode, b::GraphNode) = BroadcastedOperator(dense, x, w, b)
forward(::BroadcastedOperator{typeof(dense)}, x, w, b) = w * x .+ b
backward(::BroadcastedOperator{typeof(dense)}, x, w, b, g) = tuple(w' * g, g * x', g)

function maxpool2d() end
maxpool2d(x::GraphNode) = BroadcastedOperator(maxpool2d, x)
forward(node::BroadcastedOperator{typeof(maxpool2d)}, x) =
    let
        h, w, c = size(x)
        output = zeros(h ÷ 2, w ÷ 2, c)
        indices = CartesianIndex{3}[]
        for i in 1:c
            for j in 1:h÷2
                for k in 1:w÷2
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

backward(node::BroadcastedOperator{typeof(maxpool2d)}, x, g) =
    let
        output = zeros(size(x))
        output[node.cache] = vcat(g...)
        tuple(output)
    end