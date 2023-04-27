using PyPlot
using Printf
using Random
using MLDatasets
using ProgressBars

include("engine.jl")
include("ops.jl")
include("utils.jl")


function CNN(x, y, k1, wh, b)
    z1 = conv2d(x, k1)
    z1 = relu(z1)
    z3 = reshape(z1, Constant(26 * 26 * 16))
    z3 = relu(z3)
    z4 = dense(wh, z3)
    z4 = z4 .+ b
    loss = cross_entropy_loss(z4, y)
    graph = topological_sort(loss)
    return graph
end

function cnn_train_epoch(trainx, trainy; batch_size = 16, lr = 1e-3)
    epoch_loss = 0.0
    iter = ProgressBar(1:size(trainx, 3), printing_delay = 0.1)
    for i in iter
        x = Constant(add_dim(trainx[:, :, i]))
        y = Constant(trainy[i, :])

        graph = CNN(x, y, k1, Wh, b)
        epoch_loss += train_step!(graph)
        if i % batch_size == 0
            step!(graph, lr)
            zero_grad!(graph)
        end
        set_description(iter, string(@sprintf("Train Loss: %.3f", epoch_loss / i)))
    end
    return epoch_loss / size(trainx, 3)
end

function cnn_test_epoch(testx, testy)
    epoch_loss = 0.0
    iter = ProgressBar(1:size(testx, 3), printing_delay = 0.1)
    for i in iter
        x = Constant(add_dim(testx[:, :, i]))
        y = Constant(testy[i, :])
        graph = CNN(x, y, k1, Wh, b)
        epoch_loss += train_step!(graph)
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
k1 = Variable(create_kernel(1, 16) / 60.0);
Wh = Variable(kaiming_normal_weights(10, 26 * 26 * 16) / 60.0);
b = Variable(randn(10) / 100.0)


conv_op = NNlib.conv

for j = 1:1
    println("Epoch $j")
    perm = randperm(60_000)
    push!(
        train_loss,
        cnn_train_epoch(train_x[:, :, perm], train_y[perm, :], lr = 2e-3 / (j * 2)),
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
