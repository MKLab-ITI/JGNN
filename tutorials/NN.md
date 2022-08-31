# :zap: Your first neural network
For this example, we refer to the same dataset and experimentation 
methodology as in the in the [Basic concepts](tutorials/Introduction.md)
But we will see how to easily create a multilayer perceptron.

1. [Building layers](#building-layers)
2. [Deep architecture](#deep-architecture)

# Building layers
The class building layered architectures (`LayeredBuilder`) improves base builder
functionalities by introducing methods like `.layer(String)`. This
is an extension of normal `.operation(String)` definitions, 
with the addition that specifically the expressions `{l}` and `{l+1}` are replaced 
by the previous and current layer identifiers respectively.
Setting the input layer to `"h0"` lets it get parsed by subsequent calls.

```java
ModelBuilder modelBuilder = new LayeredBuilder("h0")
				.config("features", numFeatures)
				.config("classes", numClasses)
				.config("hidden", 64)
				.layer("h{l+1} = relu(h{l}@matrix(features, hidden)+vector(hidden))")
				.layer("yhat = softmax(h{l}@matrix(hidden, classes)+vector(classes), row)")
				.out("yhat");
```

# Deep architecture
Now that we have explained how simple layers work, let's look at two more advanced
`LayeredBuilder` methods pivotal to many deep neural networks.
The first is `.layerRepeat(String, int)`), which just repeats
the layer expression a set number of times without breaking the
functional model definition pipeline. The second is `.concat(int)`. Concatenation
is also possible with normal parsing with the `|` operation, but this performs it over any
number of layers.

We now make a more advanved model:

```java
ModelBuilder modelBuilder = new LayeredBuilder()
				.config("features", numFeatures)
				.config("classes", numClasses)
				.config("hidden", 64)
				.config("2hidden", 2*64)
				.layer("h{l+1} = relu(h{l}@matrix(features, hidden)+vector(hidden))")
				.layerRepeat("h{l+1} = relu(h{l}@matrix(hidden, hidden)+vector(hidden))", 2)
				.concat(2)
				.layer("yhat = softmax(h{l}@matrix(2hidden, classes)+vector(classes), row)")
				.out("yhat");
```

[NEXT: Your first graph neural network](tutorials/GNN.md)