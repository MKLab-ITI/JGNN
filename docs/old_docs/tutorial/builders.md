# Builders

Here we discuss the various model builders provided by JGNN to simplify model definitions.

## ModelBuilder

**When to use:** To declare architectures with a few operations.

This class is the base model builder in JGNN, providing a flexible foundation for constructing graph neural network (GNN) models. It defines models by parsing symbolic expressions that represent mathematical operations. Symbolic expressions allow to define the model's operations in a concise and readable form. Hyperparameter configurations set layer sizes and regularization rates. Neuralang Function Parsing supports parsing Neuralang functions, which are useful for defining complex operations in a single line. Input and Output Declaration makes it easy to specify the input and output of your model.

The following example shows how to create a model using this builder. The created model computes the expression `y = log(2*x + 1)` without any trainable parameters.

```java
Variable x = new Variable();
Constant c1 = new Constant(Tensor.fromDouble(1)); // Holds the constant "1"
Constant c2 = new Constant(Tensor.fromDouble(2)); // Holds the constant "2"
NNOperation mult = new Multiply()
    .addInput(x)
    .addInput(c2);
NNOperation add = new Add()
    .addInput(mult)
    .addInput(c1);
NNOperation y = new Log()
    .addInput(add);
Model model = new Model()
    .addInput(x)
    .addOutput(y);
System.out.println(model.predict(Tensor.fromDouble(2)));
```

The operation parses string expressions that are typically structured as assignments to symbols; the right-hand side of assignments accepts several operators and functions listed in the next table. Models allow multiple operations too, which are parsed through either multiple method calls or by being separated with a semicolon `;` within larger string expressions. All methods need to use previously declared symbols. For example, parsing `.out("symbol")` throws an exception if no operation previously assigned to the symbol or declared it as an input. For logic safety, **symbols cannot be overwritten or set to updated values outside of Neuralang functions**.

Finally, the base model builder class supports a roundabout declaration of Neuralang functions with expressions like this snippet taken from the Quickstart section:  
`.function("gcnlayer", "(A,h){return A@(h@matrix(?, hidden, reg))+vector(?);}")`.

In this, the first method argument is the declared function symbol's name, and the second should necessarily have the arguments enclosed in a parenthesis and the function's body enclosed in brackets. Learn more about Neuralang functions in the [next section](neuralang.md).

Model definitions have so far been too simple to be employed in practice; we need trainable parameters, which are created inline with the `matrix` and `vector` functions. There is also an equivalent Java method `ModelBuilder.param(String, Tensor)` that assigns an initialized Tensor to a variable name, but its usage is discouraged to keep model definitions simple. Additionally, there may be constants and configuration hyperparameters. Of these, constants reflect untrainable tensors and are set with `ModelBuilder.const(String, Tensor)`, whereas configuration hyperparameters are numerical values used by the parser and set with `ModelBuilder.config(String, double)`, or `ModelBuilder.config(String, String)` if the second argument value should be copied from another configuration.

Both numbers in the last snippet's symbolic definition are internally parsed into constants. On the other hand, hyperparameters can be used as arguments to dimension sizes and regularization. Retrieve previously set hyperparameters through `double ModelBuilder.getConfig(String)` or `double ModelBuilder.getConfigOrDefault(String, double)` to replace the error with a default value if the configuration is not found. The usefulness of retrieving configurations will become apparent later on.


## FastBuilder

**When to use:** To construct layered architectures.

This extends the generic `ModelBuilder` with common graph neural network operations. The main difference is that it has two constructor arguments: a square matrix `A` that is typically a normalization of the (sparse) adjacency matrix, and a feature matrix `h0`. This builder further supports the notation `symbol{l}`, where the layer counter replaces the symbol part `{l}` with 0 for the first layer, 1 for the second, and so on. Prefer the notation `h{l}` to refer to the node representation matrix of the current layer; for the first layer, this is parsed as `h0`, which is the constant set by the constructor. `FastBuilder` instances also offer a `FastBuilder.layer(String)` chain method to compute neural layer outputs. This is a variation of operation parsing, where the symbol part `{l+1}` is substituted with the next layer's counter, the expression is parsed, and the layer counter is incremented by one. Example usage is shown below, where symbolic expressions read similarly to what you would find in a paper.

```java
FastBuilder modelBuilder = new FastBuilder(adjacency, features)  // sets A, h0
    .layer("h{l+1}=relu(A@(h{l}@matrix(features, hidden, reg))+vector(hidden))")  // parses h1 = relu(A@(h0 @ ...)
    .layer("h{l+1}=A@(h{l}@matrix(hidden, classes, reg))+vector(classes)"); // parses h2 = A@(h1 @ ...)
```

Before continuing, let us give some context for the above implementation. The base operation of message passing GNNs, which are often used for node classification, is to propagate node representations to neighbors via graph edges. Then, neighbors aggregate the received representation, where aggregation typically consists of a weighted average per the normalized adjacency matrix's edge weights. For symmetric normalization, this weighted sum is compatible with spectral graph signal processing. The operation to perform one propagation can be written as `.layer("h{l+1}=A @ h{l}")`. The propagation's outcome is typically transformed further by passing through a dense layer.

In node classification settings, training data labels are typically available only for certain nodes, even though all node features are required for predictions. To retrieve the predictions for specific nodes at the top layer, you can use gather operations via brackets, or use the `FastBuilder.classify()` method, which automatically injects the required code.

For example, you can use the following snippet to gather the predictions for specific nodes. This retrieves the softmax-transformed outputs for a specified set of nodes. Alternatively, you can achieve the same outcome by chaining the `FastBuilder.classify()` method, which simplifies the process by injecting this exact code into the model definition.

```java
modelBuilder
    .var("nodes")
    .layer("h{l} = softmax(h{l})")
    .operation("output = h{l}[nodes]")
    .out("output");
```



