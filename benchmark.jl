include("engine.jl")
include("ops.jl")
include("utils.jl")

function CNN(
    x::Constant,
    y::Constant,
    k1::Variable,
    k2::Variable,
    k3::Variable,
    w1::Variable,
    b1::Variable,
    w2::Variable,
    b2::Variable,
)
    z1 = conv2d(x, k1) |> relu
    z2 = conv2d(z1, k2) |> relu
    z3 = conv2d(z2, k3) |> relu |> flatten
    z4 = dense(z3, w1, b1) |> relu
    z5 = dense(z4, w2, b2)

    loss = cross_entropy_loss(z5, y)
    graph = topological_sort(loss)
    return graph
end

## Config
conv_op = NNlib.conv
debug = false
##

k1 = Variable(create_kernel(1, 4));
k2 = Variable(create_kernel(4, 8));
k3 = Variable(create_kernel(8, 16));
w1 = Variable(kaiming_normal_weights(512, 22 * 22 * 16));
w2 = Variable(kaiming_normal_weights(10, 512));
b1 = Variable(initialize_uniform_bias(22 * 22 * 16, 512));
b2 = Variable(initialize_uniform_bias(512, 10));

function test()
    x = Constant(rand(28,28,1))
    y = Constant(rand(10))
    graph = CNN(x, y, k1, k2, k3, w1, b1, w2, b2)
    forward!(graph)
    backward!(graph)
end
@btime test()
