# :zap: Neural networks
For this example, we refer to the same dataset and experimentation 
methodology as in the in the [Learning](tutorials/Learning.md) tutorial.
But we will see how to easily create a multilayer perceptron.
We cover the following topics:

1. [Building layers](#building-layers)
2. [Deep architectures](#deep-architectures)
3. [Writing operations](#writing-operations)
4. [Save and load architectures](#save-and-load-architectures)

*Full implementations can be found in the [examples](../JGNN/src/examples/tutorial/NN.java).*

## Building layers
The class for building layered architectures (`LayeredBuilder`) improves base builder
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

## Deep architectures
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

## Writing operations
This is a good point to we present symbols you can use to define operation expressions.
Unless otherwise specified, you can replace x and y with any expression. Sometimes,
y needs to be a constant defined either by presenting a number, calling 
`ModelBuilder.config(y, double)`, or calling `ModelBuilder.constant(y, double)` to
set the numbers as hyperparameters.

|Symbol| Type | Number of inputs  |
| --- | --- | --- |
| x = y | Operator | Assign to variable x the outcome of executing y. 
| x + y | Operator | Element-by-element addition. |
| x * y | Operator | Element-by-element multiplication. |
| x - y | Operator | Element-by-element subtraction. |
| x @ y | Operator | Matrix multiplication.  |
| x | y | Operator | Row-wise concatenation of x and y. |
| x [y] | Operator | Gathers the rows with indexes y of x.| 
| transpose(x) | Function | Transposes matrix x. |
| log(x)  | Function | Apply logarithm on each tensor element. |
| relu(x) | Function | Apply relu on each tensor element. |
| tanh(x) | Function | Apply a tanh activation on each tensor element. |
| sigmoid(x) | Function | Apply a sigmoid activation on each tensor element. |
| dropout(x, y) | Function | Apply training dropout on tensor x with constant dropout rate y. |
| lrely(x, y)   | Function | Leaky relu on tensor x with constant negative slope y. |
| prelu(x)      | Function | Leaky relu on tensor x with learnanble negative slope. |
| softmax(x, y) | Function | Apply y-wide  softmax on x, where y is either row or col.|
| sum(x, y) | Function | Apply y-wide sum reduction on x, where y is either row or col.|
| max(x, y) | Function | Apply y-wide max reduction on x, where y is either row or col.|
| matrix(x, y)  | Function | Generate a matrix parameter with respective hyperparameter dimensions. |
| vector(x)     | Function | Generate a vector with respective hyperparameter size.|

Prefer using hyperparameters for matrix and vector creation, as these transfer their names to respective
dimensions for error checking. For `dropout,matrix,vector` you can also use the short names `drop,mat,vec`.

## Save and load architectures
Saving a model needs to be done via its builder. Saving stores the whole parameter
state in a specified Java path can be done per:

```java
modelBuilder.save(Paths.get("file.jgnn"));
```

A new builder (of the same type as the one that saved the model) 
can be constructed given the save Path per:

```java
modelBuilder = (LayeredBuilder)ModelBuilder.load(Paths.get("file.jgnn"));
```

You can continue working with the loaded builder, for example by adding more
layers if needed, and you can call its `.getModel()` per normal.


[NEXT: Graph neural networks](GNN.md)