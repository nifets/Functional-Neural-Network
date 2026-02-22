# Functional-Neural-Network

Functional implementation of a simple neural network in Haskell using backpropagation and stochastic gradient descent.
This was done back in 2021. I was following this material: <http://neuralnetworksanddeeplearning.com>. The entire implementation, from first principles, is done in `src/NeuralNetwork.hs`

Run XOR training:
`stack build && stack exec XorTest-exe`

Run MNIST training: (need to download MNIST data from <https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download> and put in `data` folder)
Then:
`stack build && stack exec Neural-Network-exe`

Running a `[784, 30, 10]` MLP with `20` epochs, mini-batches of `10`, learning rate `0.1` yields an accuracy of 95%.

Of course, this is using raw Haskell matrices so it's running much slower than it could be. But at least it's all pure :)

