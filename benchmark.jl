include("engine.jl")
include("ops.jl")
include("utils.jl")

function CNN(
    x::Constant,
    y::Constant,
    k1::Variable,
    k2::Variable,
    k3::Variable,
    k4::Variable,
    w1::Variable,
    b1::Variable,
    w2::Variable,
    b2::Variable,
)
    z1 = conv2d(x, k1) |> relu
    z2 = conv2d(z1, k2) |> maxpool2d |> relu
    z3 = conv2d(z2, k3) |> maxpool2d |> relu
    z4 = conv2d(z3, k4) |> maxpool2d |> relu |> flatten
    z5 = dense(z4, w1, b1) |> relu
    z6 = dense(z5, w2, b2)

    loss = cross_entropy_loss(z6, y)
    graph = topological_sort(loss)
    return graph
end

## Config
conv_op = NNlib.conv
debug = false
##

k1 = Variable(create_kernel(1, 16));
k2 = Variable(create_kernel(16, 32));
k3 = Variable(create_kernel(32, 32));
k4 = Variable(create_kernel(32, 64));

w1 = Variable(kaiming_normal_weights(128, 64));
w2 = Variable(kaiming_normal_weights(10, 128));
b1 = Variable(initialize_uniform_bias(64, 128));
b2 = Variable(initialize_uniform_bias(128, 10));

function test()
    x = Constant(rand(28,28,1))
    y = Constant(rand(10))
    graph = CNN(x, y, k1, k2, k3, k4, w1, b1, w2, b2)
    forward!(graph)
    backward!(graph)
    step!(graph, 1e-4, 1)
end
@btime test()
