# Neuralang


## Operations

Here is a table of Neuralang operations that you can use in expressions, and which can be parsed by all available builders. Standard rules for priority and parentheses apply. Prefer using configuration hyperparameters instead of direct numbers to set matrix and vector creation; configurations transfer their names to respective dimensions for error checking - more on this later.

| Symbol                       | Type      | Description                                                                                                      |
|------------------------------|-----------|------------------------------------------------------------------------------------------------------------------|
| `x = expr`                   | Operator  | Assign to variable `x` the outcome of executing expression `expr`. This expression does not evaluate to anything. |
| `x + y`                      | Operator  | Element-by-element addition.                                                                                     |
| `x * y`                      | Operator  | Element-by-element multiplication.                                                                               |
| `x - y`                      | Operator  | Element-by-element subtraction.                                                                                  |
| `x @ y`                      | Operator  | Matrix multiplication.                                                                                           |
| `x | y`                      | Operator  | Row-wise concatenation of `x` and `y`.                                                                           |
| `x [y]`                      | Operator  | Gathers the rows of `x` with indexes `y`. Indexes are still tensors, whose elements are cast to integers during this operation. |
| `transpose(A)`               | Function  | Transposes matrix `A`.                                                                                           |
| `log(x)`                     | Function  | Applies a logarithm on each element of tensor `x`.                                                               |
| `exp(x)`                     | Function  | Exponentiates each element of tensor `x`.                                                                        |
| `nexp(x)`                    | Function  | Exponentiates each non-zero element of tensor `x`. Typically used for neighbor attention (see below).             |
| `relu(x)`                    | Function  | Apply relu on each tensor element.                                                                               |
| `tanh(x)`                    | Function  | Apply a tanh activation on each tensor element.                                                                  |
| `sigmoid(x)`                 | Function  | Apply a sigmoid activation on each tensor element.                                                               |
| `dropout(x, rate)`           | Function  | Apply training dropout on tensor `x` with constant dropout rate hyperparameter `rate`.                            |
| `drop(x, rate)`              | Function  | Shorthand notation `dropout`.                                                                                    |
| `lrelu(x, slope)`            | Function  | Leaky relu on tensor `x` with constant negative slope hyperparameter `slope`.                                     |
| `prelu(x)`                   | Function  | Leaky relu on tensor `x` with learnable negative slope.                                                          |
| `softmax(x, dim)`            | Function  | Apply a softmax reduction on `x`, where `dim` is either `dim:'row'` (default) or `dim:'col'`.                     |
| `sum(x, dim)`                | Function  | Apply a sum reduction on `x`, where `dim` is either `dim:'row'` (default) or `dim:'col'`.                         |
| `mean(x, dim)`               | Function  | Apply a mean reduction on `x`, where `dim` is either `dim:'row'` (default) or `dim:'col'`.                        |
| `L1(x, dim)`                 | Function  | Apply an L1 normalization on `x` across dimension `dim`, where `dim` is either `dim:'row'` (default) or `dim:'col'`.|
| `L2(x, dim)`                 | Function  | Apply an L2 normalization on `x` across dimension `dim`, where `dim` is either `dim:'row'` (default) or `dim:'col'`.|
| `max(x, dim)`                | Function  | Apply a max reduction on `x`, where `dim` is either `dim:'row'` (default) or `dim:'col'`.                        |
| `min(x, dim)`                | Function  | Apply a min reduction on `x`, where `dim` is either `dim:'row'` (default) or `dim:'col'`.                        |
| `matrix(rows, cols)`         | Function  | Generate a matrix parameter with respective hyperparameter dimensions.                                           |
| `matrix(rows, cols, reg)`    | Function  | Generate a matrix parameter with respective hyperparameter dimensions, and L2 regularization hyperparameter `reg`.|
| `mat(rows, cols)`            | Function  | Shorthand notation `matrix`.                                                                                   |
| `mat(rows, cols, reg)`       | Function  | Shorthand notation `matrix`.                                                                                   |
| `vector(len)`                | Function  | Generate a vector with size hyperparameter `len`.                                                               |
| `vector(len, reg)`           | Function  | Generate a vector with size hyperparameter `len`, and L2 regularization hyperparameter `reg`.                    |
| `vec(len)`                   | Function  | Shorthand notation `vector`.                                                                                    |
| `vec(len, reg)`              | Function  | Shorthand notation `vector`.                                                                                    |


## Script structure


Neuralang scripts consist of functions that declare machine learning components. Use a Rust highlighter to cover all keywords. Functions correspond to machine learning modules and call each other. At their end lies a return statement, which expresses their outcome. All arguments are passed by value, i.e., any assignments are performed on fresh variable instances.

Before explaining how to use the Neuralang model builder, we present and analyze code that supports a fully functional architecture. First, look at the classify function, which for completeness is presented below. This takes two tensor inputs: nodes that correspond to identifiers indicating which nodes should be classified (the output has a number of rows equal to the number of identifiers), and a node feature matrix h. It then computes and returns a softmax for the features of the specified nodes. Aside from the main inputs, the function's signature also has several configuration values, whose defaults are indicated by a colon : (only configurations have defaults and conversely). The same notation is used to set/overwrite configurations when calling functions, as we do for softmax to apply it row-wise. Think of configurations as keyword arguments of typical programming languages, with the difference that they control hyperparameters, like dimension sizes or regularization. Write exact values for configurations, as for now no arithmetic takes place for them. For example, a configuration `patience:2*50` creates an error.

```rust
fn classify(nodes, h, epochs: !3000, patience: !100, lr: !0.01) {
    return softmax(h[nodes], dim: "row");
}
```

Exclamation marks `!` before numbers broadcast values to all subsequent function calls that have configurations with the same name. The broadcasted defaults overwrite already existing defaults of configurations with the same name anywhere in the code. All defaults are replaced by values explicitly set when calling functions. For example, take advantage of this prioritization to force output layer dimensions match your data. Importantly, broadcasted values are stored within JGNN's Neuralang model builder too; this is useful for Java integration, for example, to retrieve learning training hyperparameters from the model. To sum up, configuration values have the following priority, from strongest to weakest:

1. Arguments set during the function's call.
2. Broadcasted configurations (the last broadcasted value, including configurations set by Java).
3. Function signature defaults.

Next, let us look at some functions creating the main body of an architecture. First, `gcnlayer` accepts two parameters: an adjacency matrix `A` and input feature matrix `h`. The configuration `hidden: 64` in the function's signature specifies the default number of hidden units, whereas `reg: 0.005` is the L2 regularization applied during machine learning. The question mark ? in matrix definitions lets the autosize feature of JGNN determine dimension sizes based on a test run - if possible. Finally, the function returns the activated output of a GCN layer. Similarly, look at the gcn function. This declares the GCN architecture and has as configuration the number of output classes. The function basically consists of two `gcnlayer` layers, where the second's hidden units are set to the value of output classes. The number of classes is unknown as of writing the model, and thus is externally declared with the extern keyword to signify that this value should always be provided by Java's side of the implementation.

```rust
fn gcnlayer(A, h, hidden: 64, reg: 0.005) {
    return A@h@matrix(?, hidden, reg) + vector(hidden);
}
fn gcn(A, h, classes: extern) {
    h = gcnlayer(A, h);
    h = dropout(relu(h), 0.5);
    return gcnlayer(A, h, hidden: classes);
}
```


## Java integration

We now move to parsing our declarations with the Neuralang model builder and using them to create an architecture. To this end, save your code to a file and get it as a path `Path architecture = Paths.get("filename.nn");`, or avoid external files by inlining the definition within Java code through a multiline string per `String architecture = """ ... """;`. Below, this string is parsed within a functional programming chain, where each method call returns the modeul builder instance to continue calling more methods.

For the model builder, the following snippet sets remaining hyperparameters and overwrites the default value for `"hidden"`. It also specifies that certain variables are constants, namely the adjacency matrix A and node representation h, as well as that node identifiers are a variable that serves as the architecture's input. There could be multiple inputs, so this distinction of what is a constant and what is a variable depends mostly on which quantities change during training and is managed only by the Java side of the code. In the case of node classification, both the adjacency matrix and node features remain constant, as we work in one graph. Finally, the definition sets a Neuralang expression as the architecture's output by calling the `.out(String)` method, and applies the `.autosize(Tensor...)` method to infer hyperparameter values denoted with  a questionmark`?` from an example input. For faster completion of the model, provide a dataless list of node identifiers as input, like below.

```java
long numSamples = dataset.samples().getSlice().size();
long numClasses = dataset.labels().getCols();
ModelBuilder modelBuilder = new Neuralang()
    .parse(architecture)
    .constant("A", dataset.graph())
    .constant("h", dataset.features())
    .var("nodes")
    .config("classes", numClasses)
    .config("hidden", numClasses+2)  // custom number of hidden dimensions
    .out("classify(nodes, gcn(A,h))")  // expression to parse into a value
    .autosize(new EmptyTensor(numSamples));

System.out.println("Preferred learning rate: "+modelBuilder.getConfig("lr"));
```