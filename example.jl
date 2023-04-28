using PyPlot
using Printf
using Random
using MLDatasets
using ProgressBars

include("engine.jl")
include("ops.jl")
include("utils.jl")


function cnn_train_epoch(
    trainx::Array{Float64,3},
    trainy::Array{Float64,2};
    batch_size = 16,
    lr = 1e-3,
)
    epoch_loss = 0.0
    iter = ProgressBar(1:size(trainx, 3), printing_delay = 0.1)
    for i in iter
        x = Constant(add_dim(trainx[:, :, i]))
        y = Constant(trainy[i, :])

        graph = CNN(x, y, k1, k2, k3, w1, b1, w2, b2)
        epoch_loss += train_step!(graph)
        if i % batch_size == 0
            step!(graph, lr)
            zero_grad!(graph)
        end
        set_description(iter, string(@sprintf("Train Loss: %.3f", epoch_loss / i)))
    end
    return epoch_loss / size(trainx, 3)
end

function cnn_test_epoch(testx::Array{Float64,3}, testy::Array{Float64,2})
    epoch_loss = 0.0
    iter = ProgressBar(1:size(testx, 3), printing_delay = 0.1)
    for i in iter
        x = Constant(add_dim(testx[:, :, i]))
        y = Constant(testy[i, :])
        graph = CNN(x, y, k1, k2, k3, w1, b1, w2, b2)
        epoch_loss += test_step(graph)
        zero_grad!(graph)
        set_description(iter, string(@sprintf("Test Loss: %.3f", epoch_loss / i)))
    end
    return epoch_loss / size(testx, 3)
end

train_x, train_y = MNIST(split = :train)[:];
test_x, test_y = MNIST(split = :test)[:];

train_x = Float64.(train_x);
test_x = Float64.(test_x);

train_y = encode_one_hot(train_y);
test_y = encode_one_hot(test_y);

train_loss = [];
test_loss = [];


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


for j = 1:10
    println("Epoch $j")
    perm = randperm(100)
    push!(
        train_loss,
        cnn_train_epoch(
            train_x[:, :, perm],
            train_y[perm, :],
            lr = 2e-3 / (j),
            batch_size = 32,
        ),
    )
    push!(test_loss, cnn_test_epoch(test_x, test_y))
    println()
end

close("all")
plot(train_loss, label = "Train loss")
plot(test_loss, label = "Test loss")
xlabel("Epoch")
yscale("log")
legend()
savefig("training.png")
