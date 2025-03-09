# Models


Before looking at advanced concepts, we need to get a sense of what JGNN models look like under the hood. Models are collections of `NNOperation` instances, each representing a numerical computation with specified inputs and outputs of JGNN's `Tensor` type. Tensors will be covered later; for now, it suffices to think of them as numerical vectors, which are sometimes endowed with matrix dimensions and operations.

## Custom definition

Models consist of operations with clearly defined input and output endpoints. Broadly, operations have other operations as their own inputs and outputs. There should exist non-recurrent computational paths from inputs to outputs, and therefore operations form Directed Acyclic Graphs (DAGs - not to be confused with the graphs parsed by GNNs). More details about operations can be found in the Javadoc of the following packages:

- [nn.inputs](https://mklab-iti.github.io/JGNN/javadoc/mklab/JGNN/nn/inputs/package-summary.html)
- [nn.activations](https://mklab-iti.github.io/JGNN/javadoc/mklab/JGNN/nn/activations/package-summary.html)
- [nn.pooling](https://mklab-iti.github.io/JGNN/javadoc/mklab/JGNN/nn/pooling/package-summary.html)

As an example, create models in pure Java with a method chain like below, where the expression `y=log(2*x+1)` is implemented without any trainable parameters. After defining models, run them with the method `Tensor Model.predict(Tensor...)`, which takes as input one or more comma-separated tensors that match the model's inputs (in the same order) and computes a list of output tensors. If inputs are dynamically created, an overloaded version of the same method supports an array list of input tensors: `Tensor Model.predict(ArrayList<Tensor>)`. In both cases, a list of output tensors is returned.

```java
Variable x = new Variable();
Constant c1 = new Constant(Tensor.fromDouble(1)); // holds the constant "1"
Constant c2 = new Constant(Tensor.fromDouble(2)); // holds the constant "2"
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

## Symbolic definition

Given that several lines of code are needed to declare even simple expressions, pure Java code for creating full models tends to be cumbersome to read and maintain. JGNN therefore offers model builders that create trainable models by parsing concise expressions. There exist different kinds of builders that parse expressions of the  _Neuralang_ scripting language for matrix-based architecture definition, but differ on which programming conveniences they offer.

To begin with,`GNNBuilder` parses strings of simple Neuralang expressions. It is extended by`FastBuilder`, which provides methods that inject boilerplate code for the inputs, outputs, and layers of node classification tasks. Prefer this builder if you want to keep track of the whole model definition in one place within Java code. However, parsed expressions can not include arbitrary Neuralang function definitions, which can be declared only programmatically. Finally, the `Neuralang` builder parses all aspects of the Neuralang language, such as functional declarations of machine learning modules, where parts of function signatures manage configuration hyperparameters. Use this builder to maintain model definitions in one place (e.g., packed in one string variable, or in one file) and avoid weaving symbolic expressions in Java code.

Visit the [next tutorial](builders.md) for more details on how to use these builders. For now, let us recreate the previous example with the `ModelBuilder` class. After instantiating the builder, use a method chain to declare an input variable, parse an expression, and define the symbol `y` as an output. More details on how to create models using builders can be found in the next sections.

```java
ModelBuilder modelBuilder = new ModelBuilder()
    .var("x")
    .operation("y = log(2*x+1)")
    .out("y");
System.out.println(model.predict(Tensor.fromDouble(2)));
```

