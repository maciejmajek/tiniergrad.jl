# tiniergrad.jl
### There are many deep learning fremworks available for you to use. Need a mid sized CNN for binary classification?- Sure. Go for PyTorch. Need a huge gpt4 sized llvm? Again, go for PyTorch. This is clearly the best way! Or is it?
![image](/imgs/trends.png)
# Presenting the smallest grad so far
## [karpathy/micrograd](https://github.com/karpathy/micrograd)

### [geohot/tinygrad](https://github.com/geohot/tinygrad)

#### [maciejeg/tiniergrad.jl](https://github.com/Maciejeg/tiniergrad.jl)
## Usage
CNN + cross entropy loss
```julia
# prepare data
x = Constant(rand(28, 28, 1))
y = Constant([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
# prepare weights
k1 = Variable(create_kernel(1, 16));
k2 = Variable(create_kernel(16, 32));
k3 = Variable(create_kernel(32, 32));
k4 = Variable(create_kernel(32, 64));

w1 = Variable(kaiming_normal_weights(128, 64));
w2 = Variable(kaiming_normal_weights(10, 128));
b1 = Variable(initialize_uniform_bias(64, 128));
b2 = Variable(initialize_uniform_bias(128, 10));

# define an architecture
z1 = conv2d(x, k1) |> relu
z2 = conv2d(z1, k2) |> maxpool2d |> relu
z3 = conv2d(z2, k3) |> maxpool2d |> relu
z4 = conv2d(z3, k4) |> maxpool2d |> relu |> flatten
z5 = dense(z4, w1, b1) |> relu
z6 = dense(z5, w2, b2)
loss = cross_entropy_loss(z6, y)

# acquire graph
graph = topological_sort(loss)

# forward + backward
forward!(graph)
backward!(graph)

# Update weights
step!(graph, lr , 1)

# Pseudo batching
for i in 1:batch_size
    forward!(graph)
    backward!(graph)
end
step!(graph, lr, batch_size)
```
Disclaimer: generating the graph every iteration is not that time consuming at all!
# The choice.
### Does it matter whether one use Tensorflow, Pytorch, Jax or Flux.jl? Well, it **does**. In fact there are many differences!
# However
## The thing that matters **the most** is:
# **Do you understand the underlying math and low level operations?**
![image](/imgs/headache.png)
## **Sweating profusely**
# Do I?
## **tiniergrad.jl** is merely a proof of concept.- The concept being a **deep** understanding of even **deeper** principles of **deep** learning math.

### Hoping that one day those julia lines might help an innocent soul grasp the math behind mlp and cnn operations!

# If your head already hurts
## Here is a meme for you
![image](/imgs/meme.png)
