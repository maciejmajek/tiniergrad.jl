using PyPlot
using BenchmarkTools
abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant <: GraphNode
    output::Union{Int64,Float64,Array{Float64,N} where N}
end

mutable struct Variable <: GraphNode
    output::Array{Float64,N} where {N}
    gradient::Array{Float64,N} where {N}
    _gradient::Array{Float64,N} where {N}
    name::String
    requires_grad::Bool
    cache::Any
    Variable(output; name = "?", requires_grad = true) =
        new(output, zeros(size(output)), zeros(size(output)), name, requires_grad, nothing)
end

mutable struct ScalarOperator{F} <: Operator
    inputs::Any
    output::Any
    gradient::Any
    name::String
    cache::Any
    ScalarOperator(fun, inputs...; name = "?") =
        new{typeof(fun)}(inputs, nothing, nothing, name, nothing)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs::Any
    output::Any
    gradient::Any
    name::String
    cache::Any
    BroadcastedOperator(fun, inputs...; name = "?") =
        new{typeof(fun)}(inputs, nothing, nothing, name, nothing)
end

import Base: show, summary
show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")");
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");
show(io::IO, x::Constant) = print(io, "const ", x.output)
show(io::IO, x::Variable) = begin
    print(io, "var ", x.name)
    print(io, "\n ┣━ ^ ")
    summary(io, x.output)
    print(io, "\n ┗━ ∇ ")
    summary(io, x.gradient)
end

function visit(node::GraphNode, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end

function visit(node::Operator, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(head::GraphNode)
    visited = Set()
    order = Vector()
    visit(head, visited, order)
    return order
end

reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient .= zero(node.gradient)
reset!(node::Operator) = node.gradient = nothing

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) =
    node.output = forward(node, [input.output for input in node.inputs]...)

function zero_grad!(order::Vector)
    for node in order
        reset!(node)
    end
end

function forward!(order::Vector)
    for node in order
        compute!(node)
    end
    return last(order).output
end

update!(node::Constant, gradient) = nothing
update!(node::GraphNode, gradient) =
    let
        node.gradient = gradient
        if typeof(node) == Variable
            node._gradient += gradient
        end
    end

function backward!(order::Vector; seed = 1.0)
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end
debug = false
function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    if debug
        print(typeof(node), size(node.gradient), " => ")
    end
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)

    for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient)
        if debug
            print(size(gradient), " ")
        end
    end
    if debug
        println()
    end
    return nothing
end


function step!(graph::Vector, lr::Float64, batch_size::Int64)
    for node in graph
        if isa(node, Variable) && node.requires_grad
            node._gradient ./= batch_size
            node.output -= lr * node._gradient
            node._gradient .= 0
        end
    end
end
